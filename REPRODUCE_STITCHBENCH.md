# Reproduce MGDH on StitchBench/HD3D-Style Benchmarks

This adapter runs the imported MGDH checkpoint and renders inspectable results with a classical feature-homography canvas renderer. The method key stays `mgdh`, but the reported variant is **MGDH checkpoint + classical feature-homography canvas renderer** because the original MGDH inference code only emits a 512x512 diagnostic warp/fusion.

## Inputs

- Manifest: `D:\StitchBench_Result\_global_work\_shared\manifest.csv`
- Images: first two files in each manifest row's `image_files`
- Checkpoint: `Codes\checkpoints\model.ckpt-500000.*`
- Default output root: `D:\StitchBench_Result\mgdh`

## Environment Policy

Use the current default `python` first. The runner starts by trying the original TensorFlow import path, then automatically retries with a minimal TF1 compatibility shim when current TensorFlow lacks `tf.contrib` or other TF1 APIs.

Do not create a legacy MGDH environment unless this default-environment attempt fails beyond the compatibility shim.

## Canary

```powershell
python -m reproduce.run_stitchbench_general `
  --manifest D:\StitchBench_Result\_global_work\_shared\manifest.csv `
  --out-root D:\StitchBench_Result\mgdh `
  --scene SVA-01_chess `
  --gpu 0 `
  --force
```

Expected files:

- `D:\StitchBench_Result\mgdh\SVA-01_chess\panorama.png`
- `D:\StitchBench_Result\mgdh\SVA-01_chess\metrics.json`
- `D:\StitchBench_Result\mgdh\metrics.csv`
- `D:\StitchBench_Result\mgdh\per_pair.csv`

`panorama.png` is the inspectable stitched canvas at the source-image scale.

## Full Run

```powershell
python -m reproduce.run_stitchbench_general `
  --manifest D:\StitchBench_Result\_global_work\_shared\manifest.csv `
  --out-root D:\StitchBench_Result\mgdh `
  --gpu 0 `
  --force
```

Use `--force` when upgrading from earlier 512x512 diagnostic outputs.

## Evaluation Only

```powershell
python -m reproduce.evaluate_stitchbench_mdr_niqe `
  --manifest D:\StitchBench_Result\_global_work\_shared\manifest.csv `
  --mgdh-root D:\StitchBench_Result\mgdh `
  --output-root D:\StitchBench_Result\mgdh `
  --depth-gsp-root D:\StitchBench_Result\depth_gsp `
  --device cuda
```

`per_pair.csv` uses the shared MDR/NIQE columns consumed by MethodManagement's StitchBench aggregation tools.

## HD3D

First-pair visual gate:

```powershell
python -m reproduce.run_hd3d_style `
  --manifest D:\HD3D_Result\_global_work\_work_root\manifest.csv `
  --result-root D:\HD3D_Result `
  --pair Indoor_001_p12 `
  --gpu 0 `
  --device cuda `
  --force
```

Full run:

```powershell
python -m reproduce.run_hd3d_style `
  --manifest D:\HD3D_Result\_global_work\_work_root\manifest.csv `
  --result-root D:\HD3D_Result `
  --gpu 0 `
  --device cuda `
  --force
```

## LPS-D

First-pair visual gate uses the first row in `D:\LPS_Result\_global_work\_work_root\manifest.csv`:

```powershell
python -m reproduce.run_hd3d_style `
  --manifest D:\LPS_Result\_global_work\_work_root\manifest.csv `
  --result-root D:\LPS_Result `
  --pair Scene_01_(-10893,-1291,627)_v1_p12 `
  --gpu 0 `
  --device cuda `
  --force
```

Full run:

```powershell
python -m reproduce.run_hd3d_style `
  --manifest D:\LPS_Result\_global_work\_work_root\manifest.csv `
  --result-root D:\LPS_Result `
  --gpu 0 `
  --device cuda `
  --force
```

HD3D-style outputs are written to `<result_root>\<scene>\pair_<id>\mgdh\raw.png` with `metrics.json`, `aligned_to_gt.png`, `valid_mask.png`, and `method_status.json`.
