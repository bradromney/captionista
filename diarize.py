import os, sys, torch, json, time, subprocess, argparse
from pathlib import Path
from pyannote.audio import Pipeline
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Run pyannote speaker diarization on an input media/audio file")
parser.add_argument("input", help="Path to input media/audio file")
parser.add_argument("--device", choices=["cpu", "mps"], default=None, help="Device to run inference on (default: auto)")
args = parser.parse_args()

inp = args.input
out = Path(inp).with_suffix(".spk.json")

if not Path(inp).exists():
    print(f"❌ Input file not found: {inp}")
    sys.exit(1)

print("Starting speaker diarization...")
print(f"Input file: {inp}\nOutput file: {out}")

# --- Token handling ---
token = os.getenv("HF_TOKEN")
if token:
    print("Using HF_TOKEN from environment.")
else:
    print("No HF_TOKEN env var found. Will try Hugging Face CLI login (if you've run `huggingface-cli login`).")
    token = True  # `True` tells Pipeline to check ~/.huggingface/token

# --- Load pipeline ---
try:
    print("Loading pipeline...")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=token
    )
except Exception as e:
    print("❌ Failed to load pipeline:", e)
    print("➡ Make sure you ran `huggingface-cli login` or set HF_TOKEN, and accepted the model terms on Hugging Face.")
    sys.exit(1)

# --- Device ---
if args.device == "mps":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        print("⚠️  Requested device mps, but MPS is not available. Falling back to cpu.")
        device = torch.device("cpu")
elif args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
pipe.to(device)

# --- Process ---
print("Processing audio (this may take a while for long files)...")
start = time.time()

# Convert video to audio if necessary
audio_inp = inp
if inp.lower().endswith((".mov", ".mp4", ".avi", ".mkv")):
    audio_inp = Path(inp).with_suffix(".wav")
    if not audio_inp.exists():
        print(f"Converting {inp} → {audio_inp} …")
        cmd = [
            "ffmpeg", "-y", "-i", str(inp),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_inp)
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg and ensure it is on your PATH.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print("❌ ffmpeg failed to convert input to WAV.")
            print(f"Command: {' '.join(cmd)}")
            print(f"stderr: {e.stderr.decode(errors='ignore') if e.stderr else ''}")
            sys.exit(1)
    else:
        print(f"Using existing {audio_inp}")

dia = pipe(str(audio_inp))

timeline = []
for item in tqdm(dia.itertracks(yield_label=True), desc="Diarizing"):
    if len(item) == 3:
        seg, _track, lbl = item
        spk = str(lbl)
    else:  # fallback if label not yielded
        seg, _track = item
        spk = "SPK?"
    timeline.append({"start": seg.start, "end": seg.end, "speaker": spk})

elapsed = time.time() - start
print(f"Finished in {elapsed/60:.1f} min; segments: {len(timeline)}")

with open(out, "w") as f:
    json.dump(timeline, f)
print(f"Wrote {out}")
