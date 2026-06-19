#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Flickr30k images are distributed through Kaggle:
  https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

Download the archive manually or with the Kaggle CLI, then place/extract images
under:
  datasets/flickr30k/flickr30k_images/

The Karpathy-format captions file should be available at:
  datasets/flickr30k/caption_datasets/dataset_flickr30k.json
EOF
