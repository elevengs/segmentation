### `sam_Hn_10_03` - `vit_b_lm`

Trained on 10 high-quality manually annotated images from varying different strains and dates using the AdamW optimizer.

Metrics on three validation images:  94.35% F1 @ 50%,  74.82 F1 @ 75%, 88.29% foreground dice

### `sam_Hn_10_07_fluor`- `vit_b_lm`

Trained on randomly selected tiles (80:20 train:val split) from 13 high-quality manually annotated images from varying different strains and dates using the AdamW optimizer.
Trained to use fluorescence channel to identify cells.

Metrics on three validation images:  96.90% F1 @ 50%,  86.32% F1 @ 75%, 90.71% foreground dice

### `sam_Hn_10_08` - `vit_b_lm`

Distilled `sam_Hn_10_07_fluor` without using fluorescence; trained on 75% teacher soft outputs across >400 images with no ground truth, 25% randomly selected high-quality manually annotated tiles using the AdamW optimizer.

Metrics on three validation images:  93.79% F1 @ 50%, 74.64% F1 @ 75%, 88.23% foreground dice

Although this model appears to perform almost identically to `sam_Hn_10_03` by these metrics, its performance is actually noticeably better and more general because of its much larger training dataset.

### `sam_Hn_10_09_tiny` - `vit_t_lm`

Trained in the same manner as `sam_Hn_10_08`, but with a smaller model size.

Metrics on three validation images:  74.56% F1 @ 50%, 48.14% F1 @ 75%, 81.54% foreground dice

Despite the significantly higher F1 and dice, this model performs comparably to `sam_Hn_10_08` at roughly 3x the speed.
