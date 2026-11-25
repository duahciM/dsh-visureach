# Video observer

A DeepSeek Harness bundle for offline observation of a local video. A session may
switch videos. Each video has separate source evidence and task history.

## What it does

One video is indexed once into a keyframe catalogue. The Agent then queries
candidate frames as metadata, reads the real original frames at native
resolution, and re-decodes the source video for intervals the index covers too
sparsely. A cited original can be traced to the source SHA-256, PTS, image SHA-256
and tool receipt. Semantic support for a claim can be checked from those images.

## Tools

| Tool | Purpose | Returns images |
|---|---|---|
| `video_open` | Index a local video (cached by content hash) and bind it to the session. | no |
| `observation_goal` | Record the observation purpose, or read the session ledger. | no |
| `catalog_search` | Query frame identifiers, source times and nomination dimensions. | no |
| `wave_query` | Query navigation cues: rises, falls, plateaus, structure changes, unobserved gaps. | no |
| `observation_coverage` | Report scanned vs nominated vs actually watched, from delivery receipts. | no |
| `delivery_request` | Deliver a catalogued frame or any valid original PTS at native resolution. | yes |
| `source_resample` | Re-decode a chosen interval at a chosen density or sample count, using a cursor for further transport pages. | yes |
| `evidence_list` | List original evidence already delivered in this session and source. | no |
| `evidence_search` / `evidence_graph` / `change_search` | Retrieve nominations by root/region/relation, traverse actual 2D evidence edges, find candidate time brackets. | no |
| `visual_compare` | Run optional independent SIFT, color, quality, phase, local residual and photometry checks for two exact PTS. | no |
| `semantic_search` | Search catalogue regions with local CLIP when assets are configured. | no |
| `frame_ocr` / `video_text_search` | Check OCR at an exact PTS and query timed embedded subtitles or optional local ASR. | no |
| `observation_record` / `claim_verify` | Append a finding or revision and validate cited images by fresh original-video decode. | no |
| `evidence_crop` | Produce a labelled crop/zoom derived from a previously delivered original. | derived only |

The first image of a session requires a recorded purpose; both image tools
refuse until `observation_goal` has been set.

Install from this directory with **Plugins** > **Add plugin**, or through
`plugin_manager` `install_bundle`. See `docs/DETAIL.md` for the full behaviour,
fidelity guarantees and current limits.

## Requirements

- `ffmpeg` and `ffprobe` on `PATH`, or an explicit `ffmpegPath` setting.
- A Python interpreter with `numpy`, `opencv-python-headless` and `Pillow`.
  Point `pythonPath` at it, or leave it empty to auto-detect.
- An image-capable model route; the tools refuse to send frames otherwise.
- Configure `semanticAssets` with a directory containing `assets/text.onnx`,
  `assets/vision.onnx` and `assets/tokenizer.json`. Use `semanticPythonPaths` if
  ONNX Runtime is installed outside the interpreter. For ASR configure
  `whisperModel` with a local whisper.cpp model and an FFmpeg build that has the
  `whisper` filter. OCR uses `tesseractPath`, Tesseract on `PATH`, or Windows OCR.

## Provenance

`vendor/P2_24/` carries the frozen P2-24 research source (IF-30) unchanged, which
supplies the D1–D6 nomination, merge and catalogue logic. `python/index_video.py`
feeds an arbitrary video through that chain instead of the historical frozen
batches, and `python/navigation.py` ports the frozen p223 timeline contract into
discrete navigation cues over that catalogue.

`integration: false` disables the extension group. Any extension can also be
turned off with its own setting. See [configuration and evidence contract](docs/INTEGRATION.md).
# Offline update 0.6.0

The offline workflow now connects figure measurements, candidate views, states, original and crop receipts, and observation revisions in `evidence_routes`. See [V060](docs/V060.md) for examples, evidence boundaries and migration. The existing file-simulated live module carries over from 0.5.0.

