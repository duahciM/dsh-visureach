# Video observer

Gives an image-capable Agent a verifiable working loop over one local video:
index the file once, query keyframe metadata, read the real original frames, and
re-decode the source for intervals the index covers too sparsely.

## Tools

| Tool | Purpose | Images |
|---|---|---|
| `video_open` | Index a local video (cached by content hash) and bind it to this session. | none |
| `observation_goal` | Record the observation purpose, or read the session ledger. | none |
| `catalog_search` | Query frame identifiers, source times and nomination dimensions in a time range. | none |
| `wave_query` | Query navigation cues: count rises, falls, plateaus, structure changes and unobserved gaps. | none |
| `observation_coverage` | Report which frames were actually delivered, from receipts rather than intent. | none |
| `delivery_request` | Deliver exact catalogued frames at native resolution. | yes |
| `source_resample` | Re-decode the original video over a chosen interval and deliver evenly spaced frames. | yes |

`delivery_request` is the same name the frozen P2-24 service uses, so tool
documentation and implementation agree.

## Purpose gate

Before the first image of a task is delivered, both `delivery_request` and
`source_resample` refuse with the exact question
`请回答你观看视频的目的，这会影响后续的工具调用` and ask for `observation_goal`.

A purpose belongs to one observation task, scoped to session + video + task.
Recording a purpose opens a task, so a second question about the same video asks
again; watch history belongs to session + video, so a new task never erases what
was already seen. Reopening a video resumes its most recent task.

## Coverage

`observation_coverage` answers three separate questions over a time range:
how many frames the background scanned, how many the catalogue nominated, and
how many this agent actually watched. "Watched" means an image was really
delivered for **this video**; records are filtered by video hash, so evidence
from another clip in the same session is never counted. It reports no
whole-video unviewed ratio and never requires full coverage. Older records that
predate the video hash are reported as unattributable rather than trusted.

The `windows` zoom groups observations by the sampling interval, never by the
requested window, so two frames a second apart stay two discrete observations
even when a wide window covers both.

## Direct source access

`delivery_request` accepts any integer PTS that is a real source frame, even one
the coarse index never sampled, and labels it as off-index in its output.
`source_resample` reads any interval regardless of wave shape or keyframe
presence, so a flat wave never blocks looking at the original video.

## Navigation cues

`wave_query` reads the same catalogue and reports, for each adjacent pair of
catalogued frames, two measured axes: **I** the number of distinct region tokens
visible at that frame, and **S** the number of distinct relation structures. A
structure is keyed by region token **and** evidence family, which is what keeps
different relation structures apart; counting role strings alone collapsed them,
because in the frozen chain every D3 observation shares one role.

The frozen chain calls the second axis `R` and builds it from Figure2 graph
edges. This port carries no graph edges, so the axis is named `S` and its value
is a distinct-structure count. Every cue is a reason to look at original frames,
never an observed action, contact or cause. `wave_query` never delivers an image
and never authorises a frame read; the recorded purpose decides that.

## Ledger

Each session appends JSON lines to `<cacheDir>/sessions/<session>.jsonl`:
`video_open`, `goal`, `wave_query`, `frame_read` and `source_resample`.
`frame_read` records the observable need, the goal in force, every requested
frame and the number of images actually delivered, so "requested" stays
distinguishable from "delivered" and an interrupted run leaves a recoverable
prefix. `observation_goal` with no argument prints the current tally plus the
most recent evidence reads.

## What the metadata means

`catalog_search` reports nomination dimensions from the frozen P2-24 retrieval
chain: D1 local region, D2 local state difference, D3 image relation, D4
viewpoint, D5 unknown or conflict, D6 temporal coverage. These are low-level
image measurements. A nomination is not an object identity, an event, a contact
or a cause, and an absent nomination does not mean the thing is absent from the
video.

Frames appear in the catalogue only when at least one channel nominated them.
`catalog_search` also reports sampled-but-not-nominated frames, so "the index has
no entry" stays distinguishable from "the video has nothing there".

## Sampling density

Indexing samples at **5 fps** over the whole video, which is the frozen base
configuration (`v05/protocol.json` `sample_fps`, `if15_research` `pool.fps`).
There is no hidden frame cap: the frame budget is derived from the probed
duration, so a longer video gets proportionally more candidates instead of being
silently thinned. `video_open` reports the rate and whether any downgrade
occurred, and the catalogue rows are always the honest set that was measured.

Native PNGs are exported on demand when a frame is actually read and then
cached, so the index does not pay disk for keyframes nobody looks at. A
2-minute 720p clip indexes to roughly 11 MB and 506 catalogued frames.

If you want a different rate, pass `--fps` to `python/index_video.py` directly;
the plugin currently calls it with the default. Sampling finer than the source
rate cannot add information, and sampling coarser than 5 fps trades away exactly
the short events the D2/D5 channels are meant to catch.

## Fidelity

`frame_read` returns the original PNG at the source resolution and refuses a
frame when the attachment store normalizes it to a different size. Each result
reports the source second, integer PTS, source frame index and PNG hash prefix so
an answer can be traced back to the file. `source_resample` returns the nearest
decodable source frame to each requested second.

Between two delivered frames nothing was observed. A sampled sequence is not
evidence of a continuous process.

## Configuration

The plugin row accepts three optional settings:

- `pythonPath`: interpreter with numpy, opencv and Pillow. Empty auto-detects a
  known virtual environment, then the bundled runtime, then `python` on `PATH`.
- `cacheDir`: where per-video indexes are stored. Empty uses
  `$DSH_HOME/video-observer`.
- `ffmpegPath`: ffmpeg executable. Empty uses `ffmpeg` from `PATH`.

Indexing runs once per distinct file content; reopening the same video reuses
the cached index. Indexes are keyed by the video SHA-256, so a re-encode of the
same clip produces a separate index instead of silently reusing frames.

## Current limits

- Indexing is offline and single-threaded. A 20-second 720p clip takes about 5
  seconds; a 2-minute clip takes about 33 seconds.
- Frames are delivered only when the active model route declares image input;
  otherwise the tool refuses instead of sending a picture the model cannot read.
- Native PNGs are exported one frame at a time at read time, so the first read of
  a given frame is slower than later reads of the same frame.
- The catalogue carries no semantic labels, no object identities, no causal
  claims and no three-dimensional pose. Expect none in answers that cite it.
