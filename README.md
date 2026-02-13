# Kage Bunshin Cam

A browser web app that:

- Opens your webcam.
- Tracks both hands with MediaPipe.
- Detects approximate hand-seal heuristics for:
  - Kage Bunshin: both hands in a two-finger sign close together.
  - Chidori: one hand open with the other hand at/around that wrist.
  - Rasengan: one open palm above the other, parallel and facing each other, with a clear gap between them (top hand angled downward, bottom hand angled upward).
- Plays voice/audio for each jutsu and overlays the corresponding effect.
- Uses person segmentation so clones contain only the person; background remains
  unchanged.

## Run

1. Start a local static server from this folder:

```bash
cd /Users/kkarmakar/Documents/New\ project/kagebunshin-cam
python3 -m http.server 8080
```

2. Open http://localhost:8080 in Chrome or Edge.
3. Click **Start Camera** and allow webcam access.

## Notes

- Detection is heuristic-based (anime-inspired) and intentionally tolerant so
  near-matching seals can still trigger.
- You can tune detection thresholds in `app.js` (`isKageBunshinSeal`).
- Audio uses Web Audio + browser speech synthesis for both jutsu voices
  ("Kage Bunshin no Jutsu", "Chidori", and "Rasengan").
