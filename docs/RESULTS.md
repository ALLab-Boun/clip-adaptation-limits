# Archived Results

[← README](../README.md)

## Provenance

- **Source.** [`iremurhan/clip-retrieval-lab/results/data/`](https://github.com/iremurhan/clip-retrieval-lab/tree/main/results/data)
  in the archived MSc thesis repository, committed 2026-05-14:
  - `04A_retrieval_grouped_bar_data.csv` for retrieval. It matches
    `results/tables/04_retrieval.tex` in the same repository.
  - `01_sugarcrepe_interventions_data.csv` and
    `02_sugarcrepe_capacity_data.csv` for SugarCrepe.
- **Producer.** The thesis codebase aggregated W&B run summaries
  (`summary/test/r*` for Flickr30K, `summary/test/coco_5k_r*` for COCO,
  `summary/sugarcrepe/*`). The tables below were generated directly from those
  CSV files and rounded to one decimal.
- **Not re-run.** None of these numbers were produced or checked by running
  this repository.
- **Statistics.** Mean ± sample standard deviation across seeds. Entries
  without ± are single-seed. No significance tests or confidence intervals are
  available.
- **Labels.** Rows use this repository's registry keys. The snapshot's run ID
  is given in parentheses where it differs.

† `intra_reg` is reported in the snapshot under the run ID `B0plus`.

## Retrieval: test split

Recall@K (%). I2T is image-to-text and T2I is text-to-image.

### Flickr30K (Karpathy test, 1K images)

| Configuration | Seeds | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_min` (B0) | 3 | 91.6 ± 0.2 | 99.7 ± 0.0 | 99.8 ± 0.0 | 81.3 ± 0.1 | 96.0 ± 0.1 | 98.0 ± 0.0 |
| `intra_reg` (B0plus) † | 3 | 91.9 ± 0.2 | 99.5 ± 0.1 | 99.8 ± 0.0 | 81.6 ± 0.0 | 96.2 ± 0.1 | 98.0 ± 0.1 |
| `unfreeze_5` (B0_uf5) | 3 | 92.0 ± 0.1 | 99.6 ± 0.1 | 99.8 ± 0.1 | 81.4 ± 0.1 | 96.0 ± 0.1 | 98.1 ± 0.1 |
| `unfreeze_6` (B0_uf6) | 2 | 92.3 ± 0.4 | 99.7 ± 0.1 | 99.8 ± 0.1 | 81.6 ± 0.2 | 96.1 ± 0.0 | 98.0 ± 0.1 |
| `unfreeze_7` (B0_uf7) | 1 | 91.3 | 99.5 | 99.8 | 81.5 | 95.9 | 98.0 |
| `proj1024` (B0_proj1024) | 1 | 87.8 | 99.1 | 99.6 | 75.2 | 93.6 | 96.4 |
| `loss_siglip` (B1) | 3 | 90.3 ± 0.2 | 99.5 ± 0.3 | 99.7 ± 0.1 | 79.9 ± 0.7 | 95.3 ± 0.3 | 97.6 ± 0.3 |
| `hn_syntactic` (B2) | 3 | 91.5 ± 0.3 | 99.6 ± 0.0 | 99.8 ± 0.0 | 81.0 ± 0.3 | 95.7 ± 0.1 | 97.9 ± 0.0 |
| `seg_spatial` (B5a_seg_spatial) | 4 | 91.6 ± 0.6 | 99.4 ± 0.1 | 99.8 ± 0.1 | 81.3 ± 0.2 | 95.8 ± 0.0 | 98.0 ± 0.1 |
| `seg_semantic` (B5b_seg_semantic) | 4 | 92.1 ± 0.4 | 99.5 ± 0.1 | 99.8 ± 0.1 | 81.5 ± 0.2 | 96.0 ± 0.2 | 98.2 ± 0.1 |
| `seg_geom` (B5c_seg_continuous) | 4 | 91.9 ± 0.2 | 99.5 ± 0.1 | 99.7 ± 0.0 | 81.4 ± 0.2 | 95.9 ± 0.2 | 98.0 ± 0.0 |
| `sam_gate` (B5d_multistream_gate) | 1 | 91.0 | 99.4 | 99.7 | 80.5 | 95.7 | 97.8 |
| `sam_xattn` (B5d_multistream_crossattn) | 1 | 91.5 | 99.5 | 99.7 | 81.5 | 96.0 | 98.0 |
| `text_blip` (BLIP_TEXT) | 1 | 91.9 | 99.4 | 99.8 | 81.5 | 96.2 | 98.2 |

### MS-COCO (Karpathy test, 5K images)

| Configuration | Seeds | I2T R@1 | I2T R@5 | I2T R@10 | T2I R@1 | T2I R@5 | T2I R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base_min` (B0) | 3 | 66.0 ± 0.1 | 88.2 ± 0.1 | 93.5 ± 0.1 | 53.1 ± 0.1 | 78.8 ± 0.1 | 86.8 ± 0.0 |
| `intra_reg` (B0plus) † | 1 | 64.9 | 87.6 | 93.2 | 53.3 | 78.7 | 86.7 |
| `unfreeze_5` (B0_uf5) | 3 | 66.1 ± 0.2 | 88.3 ± 0.2 | 93.6 ± 0.1 | 53.6 ± 0.0 | 79.0 ± 0.1 | 87.0 ± 0.1 |
| `unfreeze_6` (B0_uf6) | 1 | 66.3 | 88.2 | 93.5 | 53.6 | 79.2 | 87.0 |
| `unfreeze_7` (B0_uf7) | 1 | 66.8 | 88.5 | 93.5 | 53.8 | 79.2 | 87.1 |
| `proj1024` (B0_proj1024) | 1 | 61.7 | 85.2 | 91.4 | 47.6 | 75.3 | 84.7 |
| `loss_siglip` (B1) | 3 | 64.0 ± 0.1 | 86.8 ± 0.0 | 92.4 ± 0.0 | 51.9 ± 0.0 | 77.4 ± 0.0 | 85.8 ± 0.1 |
| `hn_syntactic` (B2) | 1 | 65.1 | 87.3 | 92.7 | 52.2 | 78.0 | 86.2 |
| `aux_objcls` (B4) | 1 | 65.3 | 87.7 | 93.0 | 53.2 | 78.8 | 86.8 |
| `seg_spatial` (B5a_seg_spatial) | 1 | 64.5 | 87.1 | 93.0 | 52.5 | 78.3 | 86.4 |
| `seg_semantic` (B5b_seg_semantic) | 1 | 65.2 | 88.0 | 93.5 | 53.0 | 78.7 | 86.7 |
| `seg_geom` (B5c_seg_continuous) | 1 | 65.6 | 87.9 | 93.5 | 53.1 | 78.8 | 86.8 |

COCO values are the `coco_5k` recalls computed by `eccv-caption`, not the
`test/r*` keys. The thesis repository also holds COCO 1K, CxC and ECCV mAP@R
values for some runs (`02_coco1k_vs_eccv_mapr_data.csv`,
`02_r1_vs_eccv_mapr_data.csv`). They are not reproduced here.

## SugarCrepe

Accuracy (%), COCO-trained models. `overall` is the unweighted mean of the
seven subsets. SugarCrepe seed counts differ from the retrieval seed counts
for the same configuration.

| Configuration | Seeds | add_att | add_obj | replace_att | replace_obj | replace_rel | swap_att | swap_obj | overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_min` (B0) | 2 | 86.3 ± 0.5 | 94.5 ± 0.2 | 86.4 ± 0.2 | 97.5 ± 0.0 | 80.4 ± 0.2 | 69.7 ± 0.1 | 71.4 ± 0.0 | 83.8 ± 0.1 |
| `intra_reg` (B0plus) † | 1 | 88.2 | 94.4 | 86.0 | 97.5 | 80.7 | 69.1 | 70.6 | 83.8 |
| `unfreeze_5` (B0_uf5) | 2 | 86.4 ± 0.2 | 94.6 ± 0.2 | 87.8 ± 0.3 | 97.8 ± 0.1 | 80.3 ± 0.5 | 69.9 ± 0.5 | 69.6 ± 0.3 | 83.8 ± 0.0 |
| `proj1024` (B0_proj1024) | 1 | 81.6 | 93.1 | 81.0 | 96.4 | 77.4 | 67.4 | 70.6 | 81.1 |
| `loss_siglip` (B1) | 1 | 87.1 | 93.6 | 84.9 | 97.1 | 79.6 | 68.6 | 67.8 | 82.7 |
| `hn_syntactic` (B2) | 1 | 88.4 | 93.3 | 86.8 | 97.5 | 80.2 | 70.1 | 70.2 | 83.8 |
| `seg_spatial` (B5a_seg_spatial) | 1 | 85.8 | 93.6 | 86.7 | 97.5 | 78.7 | 68.9 | 69.8 | 83.0 |
| `seg_semantic` (B5b_seg_semantic) | 1 | 86.8 | 94.0 | 86.0 | 97.4 | 79.8 | 67.3 | 71.0 | 83.2 |
| `seg_geom` (B5c_seg_continuous) | 1 | 86.4 | 93.3 | 86.8 | 97.5 | 78.9 | 68.0 | 70.2 | 83.0 |
