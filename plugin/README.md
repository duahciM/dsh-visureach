# Video observer

A DeepSeek Harness bundle that gives an image-capable Agent a verifiable
observation loop over one local video.

## What it does

One video is indexed once into a keyframe catalogue. The Agent then queries
candidate frames as metadata, reads the real original frames at native
resolution, and re-decodes the source video for intervals the index covers too
sparsely — ending with an answer whose every claim can be traced to a source
second, a PTS and a frame hash.

## Tools

| Tool | Purpose | Returns images |
|---|---|---|
| `video_open` | Index a local video (cached by content hash) and bind it to the session. | no |
| `observation_goal` | Record the observation purpose, or read the session ledger. | no |
| `catalog_search` | Query frame identifiers, source times and nomination dimensions. | no |
| `wave_query` | Query navigation cues: rises, falls, plateaus, structure changes, unobserved gaps. | no |
| `observation_coverage` | Report scanned vs nominated vs actually watched, from delivery receipts. | no |
| `delivery_request` | Deliver exact catalogued frames at native resolution. | yes |
| `source_resample` | Re-decode the original video over a chosen interval. | yes |

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

## Provenance

`vendor/P2_24/` carries the frozen P2-24 research source (IF-30) unchanged, which
supplies the D1–D6 nomination, merge and catalogue logic. `python/index_video.py`
feeds an arbitrary video through that chain instead of the historical frozen
batches, and `python/navigation.py` ports the frozen p223 timeline contract into
discrete navigation cues over that catalogue.
