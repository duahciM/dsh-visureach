# Video observer 0.2 integration

## Install and configure

Install the folder as a local Cordis bundle through DSH Plugins or use the local
profile link for `@local/dsh-video-observer`. The `cordis.patch.yml` declares the
entry. Set `pythonPath` and `ffmpegPath` when dependencies are not available on
PATH. A release can be checked with `npm pack --dry-run` from this folder.
Reload the bundle after changing its configuration. For rollback select the
previous bundle directory. The v2 cache is separate from the v0.1 index cache.

```yaml
- id: dsh-video-observer
  config:
    sampleFps: 5
    purposeGate: true
    unobservedAxis: true
    integration: true
    denseResample: true
    evidenceSearch: true
    graphNavigation: true
    changeNavigation: true
    visualCompare: true
    semanticSearch: true
    semanticAssets: ""
    semanticPythonPaths: []
    ocr: true
    tesseractPath: ""
    textTrack: true
    whisperModel: ""
    observationRecords: true
    claimVerification: true
    derivedImages: true
    disabledDimensions: []
```

These booleans default to enabled. Unconfigured optional dependencies return a
status with no inference. `disabledDimensions` masks each D1–D6 output separately.
An explicit nonzero `frameLimit` is an opt-in coarse-analysis reduction; the default
coarse index has no fixed frame limit. Per-call image pages still have up to 8
blocks to keep transport sizes predictable, and use an explicit `next_cursor` to
continue an arbitrarily dense range. There is no cumulative viewing quota.

## Source and stored records

- The index is content-addressed with source SHA, index schema version, sampling
  settings and the optional analysis-image flag. `video_open` writes a temporary
  cache then publishes a complete index. Reopening a moved copy with identical
  bytes reuses the index without changing other sessions' bindings. Changed source
  bytes force reopening. Files with missing or non-monotonic source PTS are
  rejected. Density requests resolve real PTS and do not synthesize frames.
- The original PNG SHA and DSH attachment SHA are separate fields. The transport
  strips PNG ancillary metadata without changing its pixel data and checks the
  attachment hash and dimensions. A crop/zoom is a `derived_image` record with its
  original evidence ID, PTS, coordinates, scale and own attachment hash. It does
  not enter the watched-original axis. OCR/ASR/CLIP outputs are text metadata and
  have separate scope. ASR's timestamps are estimates on the audio time axis.
- Per-session ledgers use append-only JSONL. A fresh binding gets its own path on
  `video_open`, matching earlier goals and evidence by video SHA. Malformed lines
  do not erase usable entries. Reload or opening a video archives snapshots by
  content hash. A missing live ledger can be rebuilt from its last verified
  snapshot when it reopens. While the live ledger exists, its latest valid records
  remain available. A crash after the last archive may leave newer state in the
  live ledger only.
- `observation_record` uses `create`, `list` or `revise`, with the current revision
  ID and a reason for revision. A direct claim must reference original images
  delivered in the same session and source. `claim_verify` independently runs a
  fresh FFmpeg decode, checks PTS, SHA and dimensions, then appends an integrity
  receipt. This check leaves the semantic claim for independent visual review.

## Scope

`visual_compare` records full SIFT inliers in the ledger, observed color/edge/quality
metrics, gain/offset fit and local residuals on valid registration overlap. These
support direct local appearance hypotheses, 2D displacement and comparison, with
unassessable cases retained. Candidate grouping, true camera pose, interaction,
continuous action and 3D reconstruction remain outside the released capability.
`change_search` projects existing independent nominations into navigable intervals.
It does not reproduce the whole P2-17 rescan or automatically send images. The I/S
wave and not-watched axis never govern which source frame can be read. Long clips
still incur substantial full-source decode time and memory during indexing.
