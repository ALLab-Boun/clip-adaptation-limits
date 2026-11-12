# Lightweight CLIP Adaptation for Image–Text Retrieval

Code for a controlled comparison of lightweight fine-tuning interventions on a
pretrained CLIP dual encoder (`openai/clip-vit-large-patch14-336`) for
image–text retrieval on Flickr30K and MS-COCO. Every intervention is defined as
a small, named set of config overrides on a shared baseline. All variants run
through the same training loop, model-selection rule and evaluation suite
(in-domain retrieval, cross-dataset retrieval and SugarCrepe).

This repository is the cleaned, ViT-L/14-336 release of the code from an
archived MSc thesis repository. See [Relation to the MSc thesis repository](#relation-to-the-msc-thesis-repository).

**Status.** The code for every configuration listed below is in this
repository. This repository contains **no result files**. The numbers in
[Results](#results) are copied from a snapshot committed to the archived thesis
repository. They were produced by the thesis codebase and have **not** been
re-run with this repository.

## Contents

- [Research overview](#research-overview)
- [Relation to the MSc thesis repository](#relation-to-the-msc-thesis-repository)
- [Implemented interventions](#implemented-interventions)
- [Experimental design](#experimental-design)
- [Results](#results)
- [Key findings and limitations](#key-findings-and-limitations)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducibility notes](#reproducibility-notes)
- [Repository layout](#repository-layout)
- [License](#license)

Supplementary documents:

- [`docs/RESULTS.md`](docs/RESULTS.md): full archived result tables (R@1/5/10, SugarCrepe) with provenance.
- [`scripts/README.md`](scripts/README.md): script-by-script index.

## Research overview

**Question.** Starting from a strong pretrained CLIP model, the training budget
is fixed: 10 epochs, the projection heads and the last 4 vision blocks
configured as trainable, and the text encoder frozen. Which commonly proposed additions
change retrieval quality or compositional sensitivity, and by how much?

**Motivation.** Many fine-tuning recipes for CLIP-style retrieval combine
several changes at once: augmentation, caption rewriting, hard negatives,
auxiliary losses, extra visual inputs, alternative losses. That makes it hard
to say which component matters. This codebase isolates them. Each intervention
is a registry entry that inherits from a named parent configuration and
overrides only the keys it is meant to change. So each comparison is either
against `base_min` (minimal fine-tuning) or against `intra_reg` (the
regularized baseline that the segment-aware variants build on).

**Scope.**

- One backbone in this repository: CLIP ViT-L/14 at 336 px (HuggingFace
  Transformers, not OpenCLIP).
- Lightweight adaptation only: the text encoder stays frozen and at most the
  last 7 of 24 vision blocks are configured as trainable. The default is 4.
- Two retrieval datasets (Karpathy splits), one compositional benchmark
  (SugarCrepe), and zero-shot cross-dataset transfer.

## Relation to the MSc thesis repository

| | [`iremurhan/clip-retrieval-lab`](https://github.com/iremurhan/clip-retrieval-lab) (archived) | this repository |
|---|---|---|
| Purpose | Full working history of the MSc thesis *Cross-Modal Retrieval: Interventions, Diagnostics, and Representational Limits* (defended June 2026, archived November 2026) | Cleaned release of the ViT-L/14-336 training and evaluation code |
| Backbones | CLIP ViT-B/16 and ViT-L/14-336 | ViT-L/14-336 only |
| Extra content | SLURM launchers, MMVP-VLM evaluation, diagnostics (alignment/uniformity, missing positives, patch-level), figure scripts, `results/` snapshot, caption-rewrite generation (`legacy/paraphrase_generation/`), [methodology addendum](https://github.com/iremurhan/clip-retrieval-lab/blob/main/docs/methodology_reproducibility_addendum.md) | Not included |
| Run naming | Codenames (`B0`, `B5a`, …) | Descriptive keys (`base_min`, `seg_spatial`, …). The old codenames remain valid aliases. |
| Final code | Tag [`thesis-final`](https://github.com/iremurhan/clip-retrieval-lab/tree/thesis-final) | `main` |

What carries over and what does not:

- `src/data.py` and `src/metrics.py` are byte-identical to the thesis `main`
  branch. `src/loss.py` and `src/paraphraser.py` differ only in docstrings
  and error-message text.
- `run.py`, `src/model.py`, `src/train.py`, `src/setup.py`,
  `src/grad_cache.py`, `src/model_blip.py`, `src/eval/sugarcrepe.py`,
  `configs/config_base.yaml` and `configs/registry.yaml` differ. The changes
  are refactoring, the descriptive alias system, and the removal of ViT-B
  options, MMVP-VLM and git/run-name bookkeeping. Numerical equivalence with
  the thesis code has not been tested.
- Each registry entry has a `canonical_run_id`. It equals the thesis
  W&B run ID (for example, `seg_spatial` maps to `B5a_seg_spatial`), so runs
  from both codebases can be joined by ID.
- Use the thesis repository for the thesis figures, the CLIP ViT-B
  experiments, the caption-rewrite generation script, and the analyses not
  ported here.

## Implemented interventions

All entries live in [`configs/registry.yaml`](configs/registry.yaml) and are
selected with `--run <key>`. Each entry's *Parent* is the configuration it
inherits from. The *Changes* column lists the config keys it overrides. The
resolved settings below were checked by loading every registry entry through
`src.setup`. "Trainable" means the parameters are set to `requires_grad=True`.

### Reference configurations

| Key | Canonical ID | Parent | Changes |
|---|---|---|---|
| `unfreeze_4` | `B0_uf4` | (base config) | InfoNCE, last 4 vision blocks + post-LayerNorm + projections + logit scale trainable; intra-modal losses off; no augmented views; no paraphrases |
| `base_min` | `B0` | `unfreeze_4` | None. Same configuration as `unfreeze_4`, kept under the historical run ID `B0`. **Primary reference.** |
| `intra_reg` | `B0plus_fixed` | (base config) | `base_min` + image–image InfoNCE between two augmented views (crop scale ≥ 0.7, 2 photometric ops) + text–text InfoNCE between two precomputed LLM paraphrases (Llama-3-8B-Instruct), both with weight 1. **Reference for the segment-aware and text-encoder variants.** |

### Backbone capacity

| Key | Canonical ID | Parent | Changes |
|---|---|---|---|
| `proj_only` | `B0_projonly` | `base_min` | 0 vision blocks trainable (projections + logit scale only) |
| `unfreeze_1` … `unfreeze_3` | `B0_uf1` … `B0_uf3` | `base_min` | 1–3 trainable vision blocks |
| `unfreeze_5` … `unfreeze_7` | `B0_uf5` … `B0_uf7` | `base_min` | 5–7 trainable vision blocks |
| `proj1024` | `B0_proj1024` | `base_min` | Adds randomly initialized 768→1024 heads (Linear, BatchNorm, ReLU, Dropout) on both towers |

### Training objective and supervision

| Key | Canonical ID | Parent | Changes |
|---|---|---|---|
| `loss_siglip` | `B1` | `base_min` | Pairwise sigmoid loss (SigLIP) instead of symmetric InfoNCE. Learnable bias initialized to −10. |
| `hn_syntactic` | `B2` | `base_min` | One hard-negative caption per positive, generated online by spaCy POS-based swapping of nouns, verbs and adjectives within the batch. Image-to-text uses an N×2N logit matrix. |
| `aux_objcls` | `B4` | `base_min` | BCE auxiliary head over 80 COCO categories on the vision pooled output, weight 1.0. COCO only. |

### Segment-aware visual inputs (SAM)

| Key | Canonical ID | Parent | Changes |
|---|---|---|---|
| `seg_spatial` | `B5a_seg_spatial` | `intra_reg` | Per-patch learned embedding of the dominant SAM segment's position (3×3 grid) × size (3 bins), plus background: 28 tokens |
| `seg_semantic` | `B5b_seg_semantic` | `intra_reg` | Per-patch embedding of the dominant segment's COCO-80 category: 81 tokens. COCO: IoU matching to instance annotations. Flickr30K: CLIP zero-shot **pseudo-labels**. |
| `seg_geom` | `B5c_seg_continuous` | `intra_reg` | Per-patch 5-d geometric features of the dominant segment, passed through a 2-layer MLP |
| `sam_gate` | `B5d_multistream_gate` | `intra_reg` | Frozen, precomputed SAM ViT-B encoder features (256×8×8) fused with the CLIP CLS token by a learned gate |
| `sam_xattn` | `B5d_multistream_crossattn` | `intra_reg` | Same features, fused by cross-attention from CLS to SAM tokens |
| `sam_concat` | `B5d_multistream_concat` | `intra_reg` | Same features, fused by concatenation + MLP |
| `sam_skip` | `B5e_sam_skip` | `intra_reg` | Cross-attention residual on CLS with a sigmoid gate initialized near 0 (bias −4) |

### Diagnostic encoder swap

| Key | Canonical ID | Parent | Changes |
|---|---|---|---|
| `text_blip` | `BLIP_TEXT` | `intra_reg` | Replaces the CLIP text encoder with the frozen BLIP BERT-base text encoder (`Salesforce/blip-itm-base-coco`, unimodal mode). Only its 768→768 projection is trained. `max_length` = 512. |

Implemented but not registered: `model.unfreeze_strategy` also supports
`attention`, `mlp`, `layernorm` and `bias` (BitFit-style) partial unfreezing,
and `training.use_grad_cache` enables gradient caching. Neither is used by any
registry entry. Gradient caching requires `training.micro_batch_size`, which
is not set in the base config, and it cannot be combined with the SAM variants.

## Experimental design

### Datasets and splits

| Dataset | Use | Split |
|---|---|---|
| Flickr30K | Train / val / test | Karpathy split; `restval` merged into train; first 5 captions per image; 1K test images |
| MS-COCO 2014 | Train / val / test | Karpathy split; `restval` merged into train; first 5 captions per image; 5K test images |
| SugarCrepe | Compositional evaluation | 7 subsets (`add_att`, `add_obj`, `replace_att`, `replace_obj`, `replace_rel`, `swap_att`, `swap_obj`) on COCO val2017 images |

### Training protocol (shared by all configurations)

- Optimizer: AdamW, weight decay 0.01, batch size 128, 10 epochs.
- Learning rates and schedules:
  - Projection heads: 5e-5. Unfrozen backbone blocks: 2e-6. Both use cosine decay with a 2-epoch warmup.
  - Newly added modules (segment embeddings/MLPs, fusion modules, classifier head, 1024-d heads): 1e-3 with a constant schedule (`_CONSTANT_LR_GROUPS` in `run.py`).
- Logit scale: reset to ln(1/0.07) at initialization, then learned, with the exponentiated scale clamped at 100.
- Precision and memory: bf16 autocast on CUDA. HuggingFace gradient checkpointing is enabled whenever `model.unfreeze_vision_layers > 0`.
- Images: the image–text contrastive pair always uses the deterministic eval transform (bicubic resize + center crop to 336). Random augmentation is applied only to the two extra views used by the image–image loss.
- Seeds: Python, NumPy and PyTorch are seeded; cuDNN is deterministic; DataLoader workers get `seed + worker_id`.

### Model selection and test protocol

1. Validation retrieval runs after every epoch.
2. The selection score is **validation text-to-image R@1** (`Trainer.evaluate`).
3. `best_model.pth` is written only at save points (epochs 3, 6, 9, 10 with
   `save_freq: 3`), and only if validation R@1 improved since the last save.
   It holds the weights of that save epoch.
4. After training, the saved checkpoint is reloaded and evaluated on the
   in-domain Karpathy test split, SugarCrepe, and the other dataset's test
   split. Each of these three evaluations is independent: a failure is logged
   and does not stop the others.

### Metrics

| Evaluation | Metrics | Logged as |
|---|---|---|
| Validation (every epoch) | R@1/5/10, mAP@R, R-Precision, both directions | `val/*` |
| In-domain test | R@1/5/10, mAP@R, R-Precision, both directions | `test/*` |
| In-domain test, COCO only | COCO 1K and 5K R@K, CxC R@K, ECCV-Caption mAP@R and R-Precision (via [`eccv-caption`](https://github.com/naver-ai/eccv-caption)) | `test/coco_*`, `test/cxc_*`, `test/eccv_*` |
| Cross-dataset (COCO→Flickr30K test, Flickr30K→COCO test), no further training | Same retrieval metrics | `ood/<train>_to_<eval>/*` |
| SugarCrepe | Accuracy = fraction of triplets with sim(image, positive) > sim(image, negative), per subset, plus the unweighted mean over the 7 subsets | `sugarcrepe/*` |

Image-to-text retrieval counts a hit when any of the image's ground-truth
captions is ranked within the top K.

### Baselines and comparison structure

- `base_min` is the reference for capacity, loss and supervision changes
  (`proj_only`, `unfreeze_*`, `proj1024`, `loss_siglip`, `hn_syntactic`,
  `aux_objcls`).
- `intra_reg` is the reference for the SAM variants and `text_blip`. Compare
  those against `intra_reg`, not against `base_min`.
- A zero-shot CLIP reference is available through `tools/eval_zero_shot.py`.
  No zero-shot numbers are included in the archived snapshot.

### Comparison notes

- `intra_reg` adds several components to `base_min` together: two augmented
  image views, paraphrase pairs, and two extra loss terms.
- When the text–text loss is on, training uses only captions that have a
  precomputed rewrite (`Trainer._filter_train_dataset_for_paraphrases`). The
  number of captions excluded is logged at startup.
- On Flickr30K, `seg_semantic` uses CLIP-derived pseudo-labels, not annotations.
- `aux_objcls` is COCO-only by construction.

## Results

> **Provenance.** No results are stored in this repository, and none were
> regenerated for this documentation. The numbers below are copied from
> [`results/`](https://github.com/iremurhan/clip-retrieval-lab/tree/main/results)
> in the archived thesis repository, committed 2026-05-14. They were produced
> by the thesis codebase from W&B test summaries (`test/r*` for Flickr30K,
> `test/coco_5k_r*` for COCO). Values are the mean ± sample standard deviation
> over the listed seeds. Entries without ± come from a single seed. No
> significance tests were run.

Test R@1 (%). The full R@1/5/10 tables and SugarCrepe scores are in
[`docs/RESULTS.md`](docs/RESULTS.md).

| Configuration | Flickr30K seeds | Flickr30K I2T | Flickr30K T2I | COCO 5K seeds | COCO 5K I2T | COCO 5K T2I |
|---|---:|---:|---:|---:|---:|---:|
| `base_min` | 3 | 91.6 ± 0.2 | 81.3 ± 0.1 | 3 | 66.0 ± 0.1 | 53.1 ± 0.1 |
| `intra_reg` † | 3 | 91.9 ± 0.2 | 81.6 ± 0.0 | 1 | 64.9 | 53.3 |
| `unfreeze_5` | 3 | 92.0 ± 0.1 | 81.4 ± 0.1 | 3 | 66.1 ± 0.2 | 53.6 ± 0.0 |
| `unfreeze_6` | 2 | 92.3 ± 0.4 | 81.6 ± 0.2 | 1 | 66.3 | 53.6 |
| `unfreeze_7` | 1 | 91.3 | 81.5 | 1 | 66.8 | 53.8 |
| `proj1024` | 1 | 87.8 | 75.2 | 1 | 61.7 | 47.6 |
| `loss_siglip` | 3 | 90.3 ± 0.2 | 79.9 ± 0.7 | 3 | 64.0 ± 0.1 | 51.9 ± 0.0 |
| `hn_syntactic` | 3 | 91.5 ± 0.3 | 81.0 ± 0.3 | 1 | 65.1 | 52.2 |
| `aux_objcls` | — | — | — | 1 | 65.3 | 53.2 |
| `seg_spatial` | 4 | 91.6 ± 0.6 | 81.3 ± 0.2 | 1 | 64.5 | 52.5 |
| `seg_semantic` | 4 | 92.1 ± 0.4 | 81.5 ± 0.2 | 1 | 65.2 | 53.0 |
| `seg_geom` | 4 | 91.9 ± 0.2 | 81.4 ± 0.2 | 1 | 65.6 | 53.1 |
| `sam_gate` | 1 | 91.0 | 80.5 | — | — | — |
| `sam_xattn` | 1 | 91.5 | 81.5 | — | — | — |
| `text_blip` | 1 | 91.9 | 81.5 | — | — | — |

† Reported in the snapshot under the run ID `B0plus`.

**Not available** in the archived snapshot: `proj_only`, `unfreeze_1`–`3`,
`sam_concat`, `sam_skip`, any SAM-fusion or `text_blip` result on COCO,
cross-dataset (`ood/*`) results, zero-shot CLIP results, and mAP@R or
R-Precision tables.

## Key findings and limitations

### Observations supported by the archived snapshot

These are statements about the numbers above. They have not been re-checked
by re-running this repository.

1. **Wider randomly initialized projection heads hurt retrieval.** `proj1024`
   is below `base_min` on every reported R@K on both datasets. Its R@1 is 3.8
   to 6.1 points lower. Only one seed was run.
2. **SigLIP underperformed InfoNCE under this protocol.** `loss_siglip` is
   below `base_min` on all 12 reported R@K values (3 seeds on each dataset),
   with R@1 gaps of 1.2 to 2.1 points. This is a single hyperparameter setting:
   bias initialized to −10, sharing CLIP's reset logit scale, with no tuning.
3. **No other intervention shows a clear retrieval gain.** On Flickr30K, every
   other configuration is within 0.8 R@1 points of `base_min` in both
   directions. That is comparable to the seed standard deviations in the same
   table (up to 0.7). On COCO, most of these entries are single-seed.
4. **Swap-type compositional edits remain hardest.** These are COCO-trained
   models, mostly single-seed (see [`docs/RESULTS.md`](docs/RESULTS.md)).
   - For the six configurations without segment inputs (`base_min`,
     `intra_reg`, `unfreeze_5`, `proj1024`, `loss_siglip`, `hn_syntactic`),
     `swap_att` and `swap_obj` are the two lowest subsets (67.4–71.4%).
     `replace_obj` is the highest (96.4–97.8%).
   - Across these six, the overall SugarCrepe mean ranges from 81.1%
     (`proj1024`) to 83.8%. `hn_syntactic` (83.8%) does not exceed
     `base_min` (83.8%).

### Limitations

- **Results not reproduced here.** No configuration has been re-run with this
  repository's code. The refactored modules have not been checked for
  numerical equivalence with the thesis code.
- **Few seeds.** Many entries are single-seed, and no confidence intervals or
  paired tests are computed by the reporting code.
- **Single backbone and budget.** ViT-L/14-336, 10 epochs, frozen text
  encoder. The findings may not transfer to other backbones, longer schedules,
  or full fine-tuning.
- **Inputs not distributed.** The LLM caption rewrites (required by
  `intra_reg` and every configuration that inherits from it) and the SAM
  precomputations are not distributed. They have to be regenerated (see
  [Data preparation](#data-preparation)).

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.4.0` / `torchvision==0.19.0` to match the
Docker image. Outside Docker, install the PyTorch build for your CUDA version
first. Extra packages, depending on which configurations you run:

| Needed for | Install |
|---|---|
| `hn_syntactic` | `python -m spacy download en_core_web_sm` |
| SAM preprocessing (`seg_*`, `sam_*`) | `pip install -r requirements-preprocess.txt` (segment-anything, pycocotools) and the SAM ViT-B checkpoint `sam_vit_b_01ec64.pth` (link in the [segment-anything README](https://github.com/facebookresearch/segment-anything#model-checkpoints)) |

Docker: the experiments used `biremurhan/image-text-contrast:v0.16`, built
from [`scripts/setup/Dockerfile`](scripts/setup/Dockerfile) by
[`scripts/setup/build.sh`](scripts/setup/build.sh). The image installs
`requirements.txt` only.

## Data preparation

Configs resolve all paths relative to the repository root, under `datasets/`.

```text
datasets/
  coco/
    train2014/  val2014/  val2017/          # val2017 is used only by SugarCrepe
    annotations/                             # instances_*2014.json (aux_objcls, seg_semantic)
    caption_datasets/dataset_coco.json       # Karpathy split
    caption_rewrites_meta-llama-3-8b-instruct.json   # intra_reg family only
    coco_multilabel_map.pt                   # aux_objcls only
    sam_masks/                               # seg_* only: spatial_bins.pt, semantic_ids.pt, continuous_features.pt
    sam_encoder_features/                    # sam_* only: sam_encoder_features_pool8.pt
  flickr30k/
    flickr30k_images/*.jpg
    caption_datasets/dataset_flickr30k.json
    caption_rewrites_meta-llama-3-8b-instruct.json
    sam_masks/
    sam_encoder_features/
  sugarcrepe/*.json
```

What each group of configurations needs:

| Configurations | Images + Karpathy JSON | SugarCrepe (default-on eval) | LLM rewrites | Other |
|---|:-:|:-:|:-:|---|
| `base_min`, `unfreeze_*`, `proj_only`, `proj1024`, `loss_siglip`, `hn_syntactic` | ✓ | ✓ | | spaCy model for `hn_syntactic` |
| `aux_objcls` (COCO) | ✓ | ✓ | | `coco_multilabel_map.pt` |
| `intra_reg`, `text_blip` | ✓ | ✓ | ✓ | |
| `seg_spatial`, `seg_semantic`, `seg_geom` | ✓ | ✓ | ✓ | `sam_masks/*.pt` |
| `sam_gate`, `sam_xattn`, `sam_concat`, `sam_skip` | ✓ | ✓ | ✓ | `sam_encoder_features/` |

Steps:

1. **COCO images, annotations and Karpathy splits.** Run
   `bash scripts/setup/download_coco.sh`. It downloads COCO 2014 train/val
   images and annotations into `datasets/coco/`, and the Karpathy split files
   from [`caption_datasets.zip`](https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
   into `datasets/coco/caption_datasets/` and
   `datasets/flickr30k/caption_datasets/`. Set `DATA_ROOT` to use another
   parent directory.
2. **Flickr30K images.** Run `bash scripts/setup/download_flickr.sh`. It only
   prints instructions; the download itself is manual, from
   [Kaggle](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
   The `.jpg` files must sit directly in `datasets/flickr30k/flickr30k_images/`.
3. **SugarCrepe and COCO val2017.** Run `bash scripts/setup/download_sugarcrepe.sh`.
   It writes to `datasets/` by default and can be pointed elsewhere with
   `DATA_ROOT`.
4. **LLM caption rewrites** (the `intra_reg` family). These are not distributed
   and the generator is not in this repository. Use
   [`legacy/paraphrase_generation/generate_rewrites.py`](https://github.com/iremurhan/clip-retrieval-lab/blob/main/legacy/paraphrase_generation/generate_rewrites.py)
   from the thesis repository; its settings are recorded in the thesis
   [methodology addendum](https://github.com/iremurhan/clip-retrieval-lab/blob/main/docs/methodology_reproducibility_addendum.md).
   Then use `tools/merge_rewrite_shards.py`, `tools/check_rewrite_coverage.py`
   and `tools/clean_canonical_rewrites.py` to merge and validate the output.
   The expected format is `{sentid: [rewrite, rewrite, ...]}`.
5. **COCO multi-label targets** (`aux_objcls`):
   `python tools/build_coco_multilabel.py --ann_dir datasets/coco/annotations --output datasets/coco/coco_multilabel_map.pt`.
6. **SAM inputs** (`seg_*`, `sam_*`). Use `scripts/preprocess/precompute_sam.py`:
   - Stage 1: mask generation (shardable).
   - Stage 2: per-mode patch tensors.
   - `--encoder_features` then `--merge_encoder_features` for the fusion variants.

   The exact commands are in the script's docstring. Point `--output_dir` at
   `datasets/<dataset>/sam_masks` or `datasets/<dataset>/sam_encoder_features`
   to match the configs.

The dataset loader checks at startup that every image referenced by the split
exists, and it fails if any is missing. This also applies in debug mode,
because truncation happens after the check.

## Configuration

Configs are merged in this order. Later sources take precedence.

1. `configs/config_base.yaml`, loaded from the same directory as `--config`
2. `--config` (`configs/config_coco.yaml` or `configs/config_flickr30k.yaml`)
3. The registry entry selected by `--run`, with its `parent:` chain applied first
4. `--override key=value ...`
5. `--seed N`

Notes on `--override` value parsing:

- `true` and `false` become booleans.
- Values containing `.` become floats.
- Other numeric values become ints.
- Anything else stays a string. For example, `2e-6` stays a string; learning
  rates are cast with `float()` later, so this is harmless for them.

`--run` accepts the registry key, the `canonical_run_id`, or any listed alias.
These all resolve to the same `logging.run_id`, which names the W&B run
(`<run_id>_<dataset>_s<seed>`) and the checkpoint directory
(`checkpoints/<SLURM_JOB_ID or "local">_<run_id>_<dataset>_s<seed>/`).

## Training

W&B logging is on by default. For online logging, keep `WANDB_API_KEY` in an
env file outside the repository and source it. To run without W&B, add
`logging.use_wandb=false`.

```bash
python run.py --config configs/config_flickr30k.yaml --run base_min
python run.py --config configs/config_coco.yaml --run seg_spatial --seed 1
python run.py --config configs/config_coco.yaml --run base_min --override logging.use_wandb=false
```

Without `--run`, the run ID defaults to `unnamed`.

Smoke test (small subsets, no W&B, no post-training SugarCrepe or
cross-dataset evaluation). It still needs the full image set for the chosen
dataset.

```bash
python run.py --config configs/config_flickr30k.yaml --run base_min \
  --override debug.debug_mode=true debug.debug_samples=100 training.epochs=1 \
             logging.use_wandb=false eval.sugarcrepe.enabled=false eval.ood_retrieval.enabled=false
```

Resume. This restores the model, optimizer, scheduler, epoch counter and best
score from the checkpoint:

```bash
python run.py --config configs/config_coco.yaml --run base_min \
  --resume checkpoints/local_B0_coco_s42/best_model.pth
```

## Evaluation

- **Automatic.** At the end of `Trainer.fit`, the saved checkpoint is evaluated
  on the in-domain test split, SugarCrepe and the other dataset's test split.
  Results go to stdout and to the W&B run summary. Training writes no local
  results file.
- **SugarCrepe is on by default.** If `datasets/sugarcrepe/` or
  `datasets/coco/val2017/` is missing, the `Trainer` raises at startup. Run
  `scripts/setup/download_sugarcrepe.sh` or pass
  `eval.sugarcrepe.enabled=false`.
- **Standalone cross-dataset evaluation** of a saved checkpoint:
  `python scripts/eval/evaluate.py --checkpoint <path>/best_model.pth`
  (`--eval-dataset`, `--data-root`, `--log-wandb`; see `--help`). It writes a
  JSON cache under `artifacts/figures/cache/ood_eval/`.
- **Zero-shot CLIP reference:**
  `python tools/eval_zero_shot.py --config configs/config_flickr30k.yaml --input rgb`.
  `--input mask` feeds colorized SAM masks instead of RGB images.
- **Export W&B summaries to CSV:**
  `python scripts/util/export_wandb.py --entity <entity> --project clip-retrieval`
  writes `artifacts/wandb/runs_summary.csv`.

There is no standalone in-domain test-evaluation script in this repository.
The thesis repository has `scripts/eval/eval_in_domain_retrieval.py`.

## Reproducibility notes

- **Environment.**
  - Docker image `biremurhan/image-text-contrast:v0.16`, based on
    `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`. Most dependencies are
    pinned.
  - SAM preprocessing dependencies are listed separately in
    `requirements-preprocess.txt`.
- **Determinism.** Seeds and cuDNN flags are set (see `src/setup.py`). This
  does not guarantee bit-exact results across hardware or CUDA versions.
- **Hardware.** According to the thesis methodology addendum, training jobs
  used one GPU (NVIDIA A40 in the inspected runs), 8 CPU cores and 64 GB of
  host memory.
- **Provenance tags.**
  - `logging.lineage` is forwarded to W&B as config keys and tags. It
    currently records `aug_magnitudes: v2`, the augmentation defaults after
    they were changed from (0.4, 0.4, grayscale) to (0.7, 0.2, no grayscale).
    Bump it when a default that affects comparisons changes.
  - Checkpoints also store the full merged config.
- **Not distributed.** Datasets, LLM rewrites, SAM precomputations,
  checkpoints and W&B runs are excluded by `.gitignore` and `.dockerignore`.

## Repository layout

```text
run.py                      Training entry point: config merge, optimizer and LR groups, Trainer
configs/
  config_base.yaml          Shared defaults (model, loss, training, eval, augmentation, logging)
  config_coco.yaml          COCO paths
  config_flickr30k.yaml     Flickr30K paths
  registry.yaml             Named interventions (parent chains, aliases, canonical run IDs)
src/
  model.py                  CLIP DualEncoder, freezing, segment embeddings, SAM fusion modules
  model_blip.py             CLIP image tower + BLIP text tower (text_blip)
  loss.py                   Symmetric InfoNCE and SigLIP, intra-modal terms, hard negatives
  data.py                   Karpathy dataset, augmentation, hard-negative generator, SAM loaders
  paraphraser.py            Loader for precomputed LLM caption rewrites
  train.py                  Training loop, validation, checkpointing, post-training evaluation
  metrics.py                R@K, mAP@R, R-Precision, ECCV-Caption wrapper
  eval/sugarcrepe.py        SugarCrepe scoring
  grad_cache.py, setup.py, utils.py
scripts/                    Setup, preprocessing, standalone evaluation, W&B export (see scripts/README.md)
tools/                      Rewrite maintenance, COCO multi-label map, zero-shot evaluation
docs/                       Archived result tables
```

## License

MIT. See [`LICENSE`](LICENSE).
