# Scripts

[← README](../README.md)

Helper scripts for setup, preprocessing, standalone evaluation and result
export. Training always goes through the repository-root entry point,
`run.py`. Run every script from the repository root.

```text
scripts/
  eval/        standalone cross-dataset evaluation of a saved checkpoint
  preprocess/  SAM mask, segment-tensor and encoder-feature precomputation
  setup/       dataset downloads and Docker build
  util/        W&B summary export
```

## Training

```bash
python run.py --config configs/config_coco.yaml --run base_min
```

See the main README, sections [Configuration](../README.md#configuration) and
[Training](../README.md#training).

## Evaluation

- `eval/evaluate.py`: cross-dataset retrieval for a saved `best_model.pth`.
  It evaluates a COCO-trained checkpoint on Flickr30K test and a
  Flickr30K-trained checkpoint on COCO test, without further training.
  - It writes a JSON cache to `artifacts/figures/cache/ood_eval/`; the root can
    be changed with `CLIP_RETRIEVAL_ARTIFACT_ROOT`.
  - `--log-wandb` logs the result to the source run under `ood/*`.
  - In-domain test, SugarCrepe and cross-dataset evaluation also run
    automatically at the end of training (`src.train.Trainer`).

## Preprocessing

- `preprocess/precompute_sam.py`: precomputation for the `seg_*` and `sam_*`
  configurations. It needs the packages in `requirements-preprocess.txt` and a
  SAM ViT-B checkpoint. It has three modes:
  - Stage 1: SAM mask generation, shardable with `--shard_id` / `--num_shards`.
  - Stage 2 (`--seg_mode spatial|semantic|continuous`): per-patch tensors
    written to `--output_dir`.
  - `--encoder_features` then `--merge_encoder_features`: pooled frozen SAM
    encoder features.

  Use `datasets/<dataset>/sam_masks` and `datasets/<dataset>/sam_encoder_features`
  as output directories to match the configs. The full commands are in the
  script's docstring.

## Setup

- `setup/download_coco.sh`: downloads COCO 2014 train/val images and the 2014
  annotations into `datasets/coco/`, plus the Karpathy split files
  (`dataset_coco.json`, `dataset_flickr30k.json`) into each dataset's
  `caption_datasets/`. Override the location with `DATA_ROOT`.
- `setup/download_flickr.sh`: prints the Kaggle source for Flickr30K and the
  expected locations. It downloads nothing.
- `setup/download_sugarcrepe.sh`: downloads the seven SugarCrepe JSON files
  and COCO val2017 into `datasets/`. Override the location with `DATA_ROOT`.
- `setup/build.sh`: builds the Docker image `biremurhan/image-text-contrast:v0.16`
  for `linux/amd64` from `setup/Dockerfile`.

## Utilities

- `util/export_wandb.py`: exports the config and summary of every run in a W&B
  project to `artifacts/wandb/runs_summary.csv`. Arguments: `--entity`,
  `--project` (default `clip-retrieval`), `--out-dir`.

## Related tools (`tools/`)

- `build_coco_multilabel.py`: builds the 80-way multi-hot map used by `aux_objcls`.
- `merge_rewrite_shards.py`, `check_rewrite_coverage.py`, `clean_canonical_rewrites.py`:
  merge and validate LLM caption-rewrite files. The generator itself is in the
  archived thesis repository.
- `eval_zero_shot.py`: zero-shot CLIP retrieval (`--input rgb`) and a
  SAM-mask-only input control (`--input mask`). SugarCrepe is included by
  default.
