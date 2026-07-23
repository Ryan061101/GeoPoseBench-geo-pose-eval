# geo-pose-eval

`geo-pose-eval` is a lightweight geometric localization evaluation toolkit.
It standardizes ground-truth and model predictions into a shared Pose format, keeps the original projection logic, and evaluates visibility-aware geometric localization quality with `IoU`, `Geometry Success`, and `Overall RGS`.

This repository is intentionally kept small.
Large assets such as Matterport3D connectivity files, LandmarkRxR annotations, model predictions, and generated outputs are not committed.

## Highlights

- Unified Pose schema for reference and prediction inputs
- Visibility-first evaluation pipeline
- Preserved geometric projection behavior
- Simple command line workflow
- No third-party runtime dependency beyond Python itself

## Repository Layout

```text
Geo/
  src/geo_pose_eval/     # core implementation
  scripts/               # command line wrappers
  tests/                 # unit tests
  datasets/              # external data goes here (not committed)
  README.md
  LICENSE
  pyproject.toml
  requirements.txt
```

## What You Need To Download

Download the required external assets from their official sources:

- Matterport3D connectivity files
- LandmarkRxR annotations
- your model prediction JSON files

Place them under:

```text
datasets/
  connectivity/
  landmarkrxr/
  predictions/
```

The toolkit itself does not require RGB scans for the default Pose-based evaluation flow.
It uses connectivity JSONs to resolve `scan/viewpoint` positions.

## Installation

This project currently uses only Python standard-library modules at runtime.
`requirements.txt` installs the local package itself in editable mode.

```powershell
cd Geo
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Standard Pose Schema

All evaluation inputs are normalized into a shared Pose record:

```json
{
  "instr_id": "123",
  "scan": "scanA",
  "viewpoint": "vp1",
  "heading": 0.0,
  "elevation": 0.0,
  "x": 0.0,
  "y": 0.0,
  "z": 0.0,
  "path_id": 7,
  "language": "en-US",
  "instruction": "Walk to the lamp"
}
```

Required fields:

- `instr_id`
- `scan`
- `viewpoint`
- `heading`
- `elevation`

Optional fields:

- `x`, `y`, `z`
- `path_id`
- `language`
- `instruction`

If `x/y/z` are missing, they can be attached from connectivity files during encoding.

## Typical Workflow

### 1. Encode reference poses

```powershell
python scripts/encode_reference.py `
  datasets/landmarkrxr/LandmarkRxR_val_seen.json `
  outputs/reference_pose_val_seen.json `
  --language en-US `
  --connectivity-dir datasets/connectivity
```

### 2. Encode prediction poses

```powershell
python scripts/encode_prediction.py `
  datasets/predictions/submit_val_seen.json `
  outputs/prediction_pose_val_seen.json `
  --connectivity-dir datasets/connectivity
```

If your predictions need instruction recovery from raw annotations, add:

```powershell
--reference-json datasets/landmarkrxr/LandmarkRxR_val_seen.json
```

### 3. Run geometric evaluation

```powershell
python scripts/evaluate.py `
  outputs/reference_pose_val_seen.json `
  outputs/prediction_pose_val_seen.json `
  outputs/report_val_seen.json `
  --iou-threshold 0.5
```

## CLI Reference

### Encode Reference Poses

```powershell
python scripts/encode_reference.py INPUT_JSON OUTPUT_JSON [--language en-US] [--connectivity-dir PATH]
```

### Encode Prediction Poses

```powershell
python scripts/encode_prediction.py INPUT_JSON OUTPUT_JSON [--reference-json PATH] [--connectivity-dir PATH]
```

This supports:

- direct prediction entries with final `scan/viewpoint/heading/elevation`
- DUET-style entries that contain a per-step `poses` list

### Evaluate

```powershell
python scripts/evaluate.py REFERENCE_POSE_JSON PREDICTION_POSE_JSON OUTPUT_JSON `
  [--width 640] `
  [--height 480] `
  [--hfov 1.5707963267948966] `
  [--vfov 1.0471975511965976] `
  [--alpha-h 0.1] `
  [--alpha-v 0.1] `
  [--sigma 50.0] `
  [--iou-threshold 0.5] `
  [--landmark-distance 1.0]
```

## Metrics

Per sample:

- `visible`: true only when both reference and prediction projections are valid and have non-zero visible area
- `iou`: only computed for visible samples
- `geometry_success`: `1.0` if `iou >= iou_threshold`, else `0.0`
- `center_error`: pixel distance between projected box centers
- `geo_rgs`: `iou * visibility * exp(-center_error / sigma)`

Aggregated:

- `visibility_rate`
- `mean_iou_visible`
- `geometry_success_rate_visible`
- `mean_rgs_visible`
- `overall_rgs`

`overall_rgs` is averaged over the full evaluation set, so invisible or missing samples still lower the final result.

## Projection Notes

This release intentionally preserves the original projection behavior:

- camera pose is defined by `scan + viewpoint + heading + elevation`
- viewpoint world coordinates come from connectivity
- landmark world center is approximated from the reference pose forward direction
- bounding boxes are generated from angular radii rather than learned image priors

If you later obtain true landmark 3D centers, you can replace `landmark_center_from_reference` in `src/geo_pose_eval/projection.py` without changing the outer evaluation interface.

## Output Format

The evaluator writes a JSON report with:

- `summary`: aggregate visibility and localization metrics
- `samples`: per-instruction evaluation details

Invisible samples are explicitly tagged with statuses such as:

- `missing_prediction`
- `reference_not_projectable`
- `prediction_not_projectable`
- `reference_outside_frame`
- `prediction_outside_frame`

## Testing

```powershell
python -m unittest discover -s tests -v
```

## License

This project is released under the MIT License.
See `LICENSE`.

