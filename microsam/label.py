import os
import imageio.v3 as iio
from pathlib import Path
from micro_sam.util import get_cache_directory
from micro_sam.sam_annotator.annotator_2d import annotator_2d
import napari
from util import (
    get_device,
    make_channel_last_input,
    patch_microsam_auto_thresholds,
    get_segmentation_from_viewer,
    save_label_outputs,
)
import nd2
import argparse

parser = argparse.ArgumentParser(prog="label", usage="%(prog)s [options]")

parser.add_argument(
    "--use-fluor",
    action="store_true",
    help=(
        "Provide one fluorescence layer to the microSAM model."
    ),
)

parser.add_argument(
    "--load-labels-at",
    type=Path,
    help=(
        "Path of existing labels to load into the committed_objects layer of Napari."
    ),
)

parser.add_argument(
    "--load-labels",
    action="store_true",
    help=(
        "Load existing labels (basename of image + '_labels.png' or '_labels.tif') into the committed_objects layer of Napari. Overridden by --load-labels-at."
    ),
)

parser.add_argument(
    "-t",
    "--type",
    type=str,
    default="vit_b_lm",
    help=(
        "Model type; default: 'vit_b_lm'."
    ),
)

parser.add_argument(
    "--embedding-cache-dir",
    type=Path,
    default=Path(os.path.join(get_cache_directory(), "embeddings")),
    help=(
        "Path to store image embeddings. If this directory does not exist, it will be created; default: os.path.join(get_cache_directory(), 'embeddings')."
    ),
)
parser.add_argument(
    "--data-cache-dir",
    type=Path,
    default=Path(os.path.join(get_cache_directory(), "sample_data")),
    help=(
        "Path to cache other data besides embeddings; default: os.path.join(get_cache_directory(), 'sample_data')."
    ),
)

parser.add_argument(
    "--foreground-threshold",
    type=float,
    default=0.5,
    help=(
        "Minimum confidence level for a pixel to be marked as foreground, i.e. belonging to a cell; default: 0.5."
    ),
)

parser.add_argument(
    "--tile-side-length",
    type=int,
    default=384,
    help="Side length of a tile, which is a region to be segmented at once (in pixels); default: 384.",
)
parser.add_argument(
    "--halo-width",
    type=int,
    default=64,
    help="Width of the halo around a tile, which corresponds to the overlap between tiles (in pixels); default: 64.",
)

parser.add_argument(
    "-m",
    "--automatic-segmentation-mode",
    type=str,
    default='ais',
    help= "Method of automatic segmentation, e.g. automatic instance segmentation (AIS) or automatic prompt generation (APG); default: 'ais'.",
)

parser.add_argument(
    "checkpoint",
    type=Path,
    help="Path of microSAM checkpoint to use for segmentation."
)

parser.add_argument(
    "img",
    type=Path,
    help="Path to the image to label. May be an ND2 file or any image readable by imageio.imread.",
)


def main() -> None:
    args = parser.parse_args()
    os.makedirs(args.embedding_cache_dir, exist_ok=True)
    os.makedirs(args.data_cache_dir, exist_ok=True)
    os.environ["MICROSAM_CACHEDIR"] = str(args.data_cache_dir)

    stem = str(args.img.with_suffix(""))
    if args.img.suffix.lower() == ".nd2":
        raw = nd2.imread(args.img)
    else:
        raw = iio.imread(args.img)
    print(f"Loaded image shape: {raw.shape}, dtype: {raw.dtype}")

    if raw.ndim != 3 or raw.shape[0] < 2:
        raise ValueError(
            "Expected image shape (2, Y, X) with phase in raw[0] and fluorescence in raw[1]. "
            f"Got shape {raw.shape}."
        )

    phase = raw[0]
    fluor = raw[1]

    labels = None
    if args.load_labels_at is not None:
        labels = iio.imread(args.load_labels_at)
    elif args.load_labels:
        labels_path = Path(stem + "_labels.png")
        if not labels_path.is_file():
            labels_path = Path(stem + "_labels.tif")
        labels = iio.imread(labels_path)

    img_for_sam = make_channel_last_input(phase, fluor, args.use_fluor)
    print(f"Input passed to microSAM: {img_for_sam.shape}, dtype: {img_for_sam.dtype}")

    device = get_device()
    print(f"Using device: {device}")

    viewer = napari.Viewer()

    viewer.add_image(phase, name="phase", visible=True)
    viewer.add_image(fluor, name="fluorescence", visible=False)

    patch_microsam_auto_thresholds(
        foreground_threshold=args.foreground_threshold,
        labels_threshold=None,
        force_mode=args.automatic_segmentation_mode,
    )

    annotator_2d(
        img_for_sam,
        model_type=args.type,
        tile_shape=(args.tile_side_length, args.tile_side_length),
        halo=(args.halo_width, args.halo_width),
        checkpoint_path=str(args.checkpoint),
        embedding_path=str(args.embedding_cache_dir / (args.img.stem + ".zarr")),
        device=device,
        prefer_decoder=True,
        segmentation_result=labels,
        viewer=viewer,
        return_viewer=True,
    )

    napari.run()

    segmentation = get_segmentation_from_viewer(viewer)

    if segmentation is None:
        print("No labels layer found after closing napari; no labels was saved.")
        return

    print(
        f"Saving segmentation with shape {segmentation.shape}, dtype {segmentation.dtype}"
    )
    save_label_outputs(segmentation, phase, fluor, stem)
    print("Saved labels and overlay outputs.")


if __name__ == "__main__":
    main()
