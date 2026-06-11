import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wav
import tempfile
import os
from datetime import datetime

SAMPLE_RATE = 16000
MODEL_SIZE = "base"

print(f"Loading Whisper model ({MODEL_SIZE})...")
model = whisper.load_model(MODEL_SIZE)
print("Ready.\n")


def record_until_enter() -> np.ndarray:
    print("Recording... press Enter to stop.")
    chunks = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()

    return np.concatenate(chunks, axis=0)


def transcribe(audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    wav.write(tmp_path, SAMPLE_RATE, audio)
    result = model.transcribe(tmp_path, fp16=False)
    os.remove(tmp_path)
    return result["text"].strip()


def save_transcript(text: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"transcripts/{timestamp}.txt"
    os.makedirs("transcripts", exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(text)
    return filename


def main():
    while True:
        print("\nPress Enter to start recording (or Ctrl+C to quit).")
        try:
            input()
        except KeyboardInterrupt:
            print("\nDone.")
            break

        audio = record_until_enter()
        print("Transcribing...")
        text = transcribe(audio)

        print(f"\nTranscript:\n{text}\n")

        saved = save_transcript(text)
        print(f"Saved to {saved}")


if __name__ == "__main__":
    main()
