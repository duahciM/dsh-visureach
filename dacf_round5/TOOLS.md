# VisuReach-AMR 0.6.0-r5.0

This isolated study bundle contains the frozen VisuReach 0.6.0 source and adds a study configuration. A0 serves all offline tools. A1 adds a flat list. A2 adds the hierarchy, native refinement and `evidence_routes.amr_node_id`. The programme and source live in the local study workspace. Configure `planPath`, `authorizedSource`, `pythonPath`, a distinct `cacheDir`, `arm`, and optionally `automaticPolicy: S1` in the Codex/DSH local profile. S0 is the candidate study preset. Neither policy runs developer-side actions automatically or limits the observer. The plan must match the source SHA.

| Action | Example |
|---|---|
| Inspect the index | `amr_query({start:5,end:9,page_size:8})` |
| Request native waveform | `amr_query({node_id:"source",detail:"wave"})` |
| Query measured and recorded evidence | `evidence_routes({amr_node_id:"source",channel:"computed"})` |
| Refine a time node | `amr_refine({node_id:"source",mode:"time",reason:"Check evidence in an uncertain interval."})` |
| Refine a local region | `amr_refine({node_id:"source",mode:"space",pts:[0,5120],native_xywh:[75,74,100,71],crop_pts:5120,reason:"Inspect a local feature on verified originals."})` |
| Check exact evidence | `evidence_reference({ids:["<original or derived receipt>"]})`, `claim_verify({evidence_ids:["<receipt>"]})` |

Use source seconds for ranges and actual integer PTS for exact reads. Use a returned node ID, an applicable ROI in pixels of the original, and real source PTS inside that node. `detail:"proof"` opens computation references. `evidence_detail` retrieves the underlying native comparison including the held-out validation and applicability. For an unevaluable mapping keep the domain unknown. Use ordinary `delivery_request` or `source_resample` with any legal source interval. A stop describes an automatic policy; an unevaluated node says `not_evaluated`. Node depth does not establish evidentiary strength. Session children retain source receipts and parent links in the live session and are logged for audit. The frozen initial tree is read-only.
