# Speech to Text

## What it does

Records audio from your microphone, transcribes it locally using OpenAI's Whisper model, and saves the result as a timestamped `.txt` file in the `transcripts/` folder.

## Pipeline

```
Microphone input
    ↓  sounddevice (captures raw audio at 16kHz)
NumPy audio array
    ↓  Whisper (base model, runs locally)
Transcribed text
    ↓  saved to transcripts/YYYY-MM-DD_HH-MM-SS.txt
```

## How to use

```bash
pip install openai-whisper sounddevice scipy numpy
python speech_to_text.py
```

1. Press **Enter** to start recording
2. Speak
3. Press **Enter** again to stop
4. Whisper transcribes the audio
5. Transcript is printed and saved to `transcripts/`
6. Repeat or **Ctrl+C** to quit

## Key decisions

**Whisper `base` model** — good balance of speed and accuracy for conversational speech. Runs entirely on your machine; no data leaves your computer and no API cost. You can swap to `small` or `medium` in `MODEL_SIZE` for better accuracy at the cost of speed.

**16kHz sample rate** — Whisper was trained on 16kHz audio, so recording at this rate avoids any resampling step.

**Temp WAV file** — Whisper expects a file path, not a raw array, so the audio is written to a temporary file, transcribed, then immediately deleted.

**Timestamped filenames** — each session gets its own file (`2026-06-04_14-32-01.txt`), making it easy to track and ingest conversations individually into the RAG pipeline later.

## What comes next

The saved transcripts feed directly into the ingestion step — each `.txt` file gets chunked, embedded, and stored in ChromaDB so you can query any past conversation later.
