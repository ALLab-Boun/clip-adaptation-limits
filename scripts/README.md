# Scripts

This directory contains lightweight helpers for the paper-release archive.

```text
scripts/
  eval/        standalone cross-dataset evaluation helper
  preprocess/ SAM mask/segment/encoder-feature preprocessing
  setup/      dataset, Docker, and one-time setup helpers
  util/       WandB export helper
```

## Training

Training uses the repository root entry point:

```bash
python run.py --config configs/config_coco.yaml --run base_min
```

## Evaluation

- `eval/evaluate.py`: standalone cross-dataset retrieval evaluation for a saved
  checkpoint. Training-time SugarCrepe and OOD retrieval are handled inside
  `src.train.Trainer`.

## Preprocessing

- `preprocess/precompute_sam.py`: SAM mask generation, segment tensors, and
  frozen SAM encoder-feature extraction.

## Setup

- `setup/download_coco.sh`: download COCO 2014 images and annotations.
- `setup/download_flickr.sh`: print the Flickr30k Kaggle source and target
  location notes.
- `setup/download_sugarcrepe.sh`: download SugarCrepe JSONs and COCO val2017.
- `setup/build.sh`: build the project Docker image.

## Utilities

- `util/export_wandb.py`: export WandB run summaries.
