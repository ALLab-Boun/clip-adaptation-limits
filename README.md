# CLIP Retrieval Lab

Code for CLIP-based image-text retrieval experiments on Flickr30k and MS-COCO.
The repository contains the training code, ablation registry, evaluation scripts,
and preprocessing utilities used for the paper experiments.

The code supports:

- CLIP dual-encoder fine-tuning with configurable vision-layer unfreezing.
- InfoNCE and SigLIP losses.
- Intra-modal image/text regularization with augmented image views and
  precomputed caption rewrites.
- Syntactic hard-negative captions.
- COCO object-class auxiliary supervision.
- Segment-aware and SAM-feature fusion ablations.
- In-domain retrieval, cross-dataset retrieval, and SugarCrepe evaluation.

## Requirements

- Python 3.9+.
- Python packages from `requirements.txt`.

`requirements.txt` pins the PyTorch version used by the Docker image. If you
need a CUDA-specific wheel outside Docker, install the matching PyTorch build
for your platform before or after installing the remaining requirements.

```bash
pip install -r requirements.txt
```

For the syntactic hard-negative ablation, install the spaCy English model in the
runtime environment:

```bash
python -m spacy download en_core_web_sm
```

## Repository Layout

```text
.
├── configs/          # Base config, dataset configs, and ablation registry
├── src/              # Training, model, data, loss, metrics, and eval cores
├── scripts/          # Evaluation, preprocessing, export, and setup helpers
├── tools/            # Dataset/rewrite preparation and zero-shot utilities
├── run.py            # Training entry point
└── requirements.txt
```

## Data Layout

Configs expect datasets under `datasets/` by default. Large data files are not
included in the release archive.

```text
datasets/
  coco/
    train2014/
    val2014/
    caption_datasets/dataset_coco.json
    caption_rewrites_*.json
    sam_segments/
    sam_encoder_features/
    coco_multilabel_map.pt
  flickr30k/
    flickr30k_images/
    caption_datasets/dataset_flickr30k.json
    caption_rewrites_*.json
    sam_segments/
    sam_encoder_features/
  sugarcrepe/
```

Download helpers are provided in `scripts/setup/download_coco.sh`,
`scripts/setup/download_flickr.sh`, and `scripts/setup/download_sugarcrepe.sh`.

## Configuration

Training configs are merged in this order:

1. `configs/config_base.yaml`
2. `configs/config_coco.yaml` or `configs/config_flickr30k.yaml`
3. Optional ablation entry from `configs/registry.yaml` via `--run`
4. CLI overrides via `--override key=value`

Examples:

```bash
python run.py --config configs/config_coco.yaml
python run.py --config configs/config_flickr30k.yaml --run base_min
python run.py --config configs/config_coco.yaml --run seg_spatial --seed 1
python run.py --config configs/config_flickr30k.yaml --override training.epochs=10
```

Common registry keys include `base_min`, `intra_reg`, `loss_siglip`,
`hn_syntactic`, `aux_objcls`, `seg_spatial`, `seg_semantic`, `seg_geom`,
`sam_gate`, `sam_xattn`, and `sam_skip`.

## Training

Local training:

```bash
python run.py --config configs/config_coco.yaml --run base_min
python run.py --config configs/config_flickr30k.yaml --run intra_reg
```

Resume from a checkpoint:

```bash
python run.py --config configs/config_coco.yaml --run base_min --resume checkpoints/best_model.pth
```

Fast debug run:

```bash
python run.py \
  --config configs/config_flickr30k.yaml \
  --override debug.debug_mode=true debug.debug_samples=100
```

## Evaluation

Post-training SugarCrepe and cross-dataset retrieval are run by
`src.train.Trainer` according to the `eval:` section of `configs/config_base.yaml`.
The only standalone evaluation helper kept in `scripts/eval/` is the
cross-dataset retrieval CLI:

```bash
python scripts/eval/evaluate.py --help
```

The lightweight WandB summary exporter is in `scripts/util/`.

For a script-by-script index, see `scripts/README.md`.

## Preprocessing

COCO multilabel targets:

```bash
python tools/build_coco_multilabel.py --help
```

Caption rewrite maintenance:

```bash
python tools/check_rewrite_coverage.py --help
python tools/clean_canonical_rewrites.py --help
python tools/merge_rewrite_shards.py --help
```

Zero-shot CLIP retrieval utility:

```bash
python tools/eval_zero_shot.py --help
```

SAM segment and encoder-feature preprocessing is implemented in
`scripts/preprocess/precompute_sam.py`.

## WandB Notes

For online WandB logging, create an env file outside the repository, for example:

```text
WANDB_API_KEY=your_key_here
```

Then source it in your runtime environment. Do not commit local env files or API
keys.

The Docker image used for the consolidated experiments is
`biremurhan/image-text-contrast:v0.16`. The Dockerfile is in
`scripts/setup/Dockerfile`.

## Release Hygiene

The repository is intended to ship source code, configs, scripts, and lightweight
analysis utilities only. Datasets, checkpoints, WandB runs, logs, generated
figures, result tables, and local virtual environments are excluded by
`.gitignore` and `.dockerignore`.
