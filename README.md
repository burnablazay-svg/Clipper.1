# Clipper

Standalone clipping agent. Paste a link, get clips. No Everygen account needed.

## Setup
1. Install ffmpeg (`brew install ffmpeg` or `apt install ffmpeg`).
2. `pip install -r requirements.txt`
3. `export ANTHROPIC_API_KEY=your_key`

## Use
    python clipper.py "https://youtu.be/THRBZfHK8CQ" --count 6 --vertical --captions
    python clipper.py URL --focus "the funniest reactions"

Output goes to `clips/`: `clip01.mp4`, `clip01.srt`, and `clips.json`.

## How it picks moments
YouTube's most-replayed heatmap (when the video has one) plus a Whisper
transcript go to Claude, which scores moments using a setup, twist, peak,
hard-cut rubric. Set `CLIPPER_MODEL` to change the Claude model.

## Web interface
    python app.py
Open http://127.0.0.1:5000, paste a link, pick options, and click Find clips.
Clips appear as they finish and are saved under `jobs/<id>/`. It only listens
on your own machine (127.0.0.1).
