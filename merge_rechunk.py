import json, sys, srt, datetime as dt, re
from pathlib import Path

# ---------- Tunables ----------
MIN_DUR = 1.5             # minimum seconds per caption (merge if shorter unless forced)
MAX_DUR = 6.5             # maximum seconds per caption
MAX_CHARS = 84            # max total chars (≈2 lines)
MAX_CPS = 15.0            # characters per second cap
PAUSE_SOFT = 0.35         # prefer split if >= soft pause
PAUSE_HARD = 0.60         # force split if >= hard pause
MAX_WORDS_PER_LINE = 8    # max words per line
POST_PUNCT_MIN_CHARS = 12 # avoid splitting if next chunk would be shorter than this
POST_PUNCT_MIN_WORDS = 3  # avoid splitting if next chunk would be too few words
LONG_PAUSE_RETAG = 10.0   # relabel speaker after silence ≥ this many seconds
ADD_ELLIPSIS_ON_PAUSE = True
# ------------------------------

# Speaker mapping (names instead of codes)
SPEAKER_MAP = {"SPEAKER_00": "Dave", "SPEAKER_01": "Sue", "SPK0": "Dave", "SPK1": "Sue"}

END_PUNCT = re.compile(r'[.!?…]+$')

ABBREV = re.compile(r'^(mr|mrs|ms|dr|st|sr|jr|vs|etc|e\.g|i\.e|u\.s)\.?$', re.I)
NUMTOKEN = re.compile(r'^\$?[\d][\d,.\-/]*%?$')

def is_bad_split_token(t):
    return bool(ABBREV.match(t)) or bool(NUMTOKEN.match(t))

def load_words(whisper_json_path):
    data = json.load(open(whisper_json_path))
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words") or []:
            if "start" in w and "end" in w and "word" in w:
                words.append({"start": float(w["start"]), "end": float(w["end"]), "text": w["word"].strip()})
    # fallback if no word-level timestamps
    if not words:
        for seg in data.get("segments", []):
            words.append({"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()})
    return words

def load_spk(spk_json_path):
    # Expect list of {start, end, speaker}
    segs = json.load(open(spk_json_path))
    segs.sort(key=lambda x: x["start"])
    return segs

def assign_speakers(words, spk):
    out = []
    if not spk:
        # No diarization available; default speaker tag
        return [{**w, "spk": "SPK0"} for w in words]
    i = 0
    for w in words:
        # advance speaker pointer until w falls inside or before segment
        while i+1 < len(spk) and w["start"] >= spk[i+1]["start"]:
            i += 1
        cur = spk[i]
        if w["start"] > cur["end"] and i+1 < len(spk):
            i += 1
            cur = spk[i]
        out.append({**w, "spk": cur.get("speaker", "SPK0")})
    return out

def cps_ok(chars, dur): return dur > 0 and (chars / dur) <= MAX_CPS

def wrap_lines(tokens):
    # simple 1–2 line balance by words
    if len(tokens) <= MAX_WORDS_PER_LINE:
        return " ".join(tokens)
    best_txt, best_score = None, 1e9
    for k in range(1, len(tokens)):
        l1, l2 = " ".join(tokens[:k]), " ".join(tokens[k:])
        if max(len(l1), len(l2)) > MAX_CHARS: continue
        if len(l1.split()) > MAX_WORDS_PER_LINE or len(l2.split()) > MAX_WORDS_PER_LINE: continue

        # --- scoring tweak: prefer breaks at punctuation ---
        balance = abs(len(l1) - len(l2))
        punct_bonus = -2 if END_PUNCT.search(l1) else 0  # reward if line1 ends with . ! ? …
        if re.search(r'[,:;—-]$', l1): punct_bonus -= 1
        score = balance + (0 if END_PUNCT.search(l1) else 3) + punct_bonus
        if score < best_score:
            best_score, best_txt = score, l1 + "\n" + l2
    return best_txt or " ".join(tokens)

def rechunk(words):
    subs, idx = [], 1
    if not words: return subs
    cur = [words[0]]
    start = words[0]["start"]
    last = words[0]
    cur_spk = words[0]["spk"]

    for w in words[1:]:
        gap = w["start"] - last["end"]
        dur = w["end"] - start
        same_spk = (w["spk"] == cur_spk)

        tokens_try = [x["text"] for x in cur] + [w["text"]]
        text_try = " ".join(tokens_try)
        dur_try = w["end"] - start  

        force = (
            (dur_try > MAX_DUR) or
            (len(text_try) > MAX_CHARS) or
            (not cps_ok(len(text_try), dur_try)) or
            (not same_spk)
        )
        prefer = (gap >= PAUSE_HARD) or (gap >= PAUSE_SOFT and END_PUNCT.search(last["text"]))

        # Abbrev/number guard (avoid splitting after e.g., U.S., Dr., 3.5%)
        if is_bad_split_token(last["text"].lower()):
            prefer = False

        # --- NEW: avoid premature split after punctuation if next chunk would be tiny ---
        if END_PUNCT.search(last["text"]) and gap >= PAUSE_SOFT:
            next_len = len(w["text"])
            if next_len < POST_PUNCT_MIN_CHARS or len(w["text"].split()) < POST_PUNCT_MIN_WORDS:
                prefer = False

        # --- NEW: force a relabel after long silence even if same speaker ---
        if same_spk and gap >= LONG_PAUSE_RETAG:
            same_spk = False
            force = True

        if force or prefer:
            content = wrap_lines([x["text"] for x in cur])
            
            # --- NEW: enforce min duration (merge if too short unless forced) ---
            cur_dur = cur[-1]["end"] - start
            cur_chars = len(" ".join(x["text"] for x in cur))
            if (cur_dur < MIN_DUR or cur_chars < POST_PUNCT_MIN_CHARS) and same_spk:
                # Don't flush yet, just keep accumulating
                cur.append(w)
                last = w
                continue

            # Map speaker to name
            label = SPEAKER_MAP.get(cur_spk, cur_spk)
            header = f">> {label}"
            
            # Only add header when speaker changes or after long pause
            should_add_header = (
                len(subs) == 0 or 
                subs[-1].content.split("\n", 1)[0] != header or
                not same_spk
            )
            
            final_content = f"{header}\n{content}" if should_add_header else content
            
            subs.append(srt.Subtitle(
                index=idx,
                start=dt.timedelta(seconds=start),
                end=dt.timedelta(seconds=cur[-1]["end"]),
                content=final_content
            ))
            
            # Add ellipsis to previous subtitle if splitting on pause
            if ADD_ELLIPSIS_ON_PAUSE and gap >= PAUSE_SOFT and subs and len(subs) > 1:
                prev_sub = subs[-2]
                last_lines = prev_sub.content.split("\n")
                if not END_PUNCT.search(last_lines[-1]):
                    subs[-2].content = subs[-2].content.rstrip() + " …"
            
            idx += 1
            # new chunk
            cur = [w]  
            start = w["start"]
            cur_spk = w["spk"]
        else:
            cur.append(w)
        last = w

    # flush last
    content = wrap_lines([x["text"] for x in cur])
    
    # Map speaker to name
    label = SPEAKER_MAP.get(cur_spk, cur_spk)
    header = f">> {label}"
    
    # Only add header when speaker changes or after long pause
    should_add_header = (
        len(subs) == 0 or 
        subs[-1].content.split("\n", 1)[0] != header
    )
    
    final_content = f"{header}\n{content}" if should_add_header else content
    
    subs.append(srt.Subtitle(
        index=idx,
        start=dt.timedelta(seconds=start),
        end=dt.timedelta(seconds=cur[-1]["end"]),
        content=final_content
    ))
    # reindex
    for i, s in enumerate(subs, 1): s.index = i
    return subs

def main(whisper_json, spk_json, out_srt):
    words = load_words(whisper_json)
    spk = load_spk(spk_json)
    words = assign_speakers(words, spk)
    subs = rechunk(words)
    with open(out_srt, "w") as f:
        f.write(srt.compose(subs))
    print(f"Wrote {out_srt}  ({len(subs)} subtitles)")
    
    # Also write VTT version
    out_vtt = Path(out_srt).with_suffix(".vtt")
    with open(out_vtt, "w") as f:
        f.write("WEBVTT\n\n")
        f.write(srt.compose(subs))
    print(f"Wrote {out_vtt}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 merge_rechunk.py IMG_7744.json IMG_7744.spk.json IMG_7744_tidy.srt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
