import napari
import numpy as np
import torch
from PIL import Image
import time

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def make_channel_last_input(
    phase: np.ndarray, fluor: np.ndarray, use_fluor: bool
) -> np.ndarray:
    """
    micro_sam's 2D annotator expects either:
      - a 2D image with shape (Y, X), or
      - a channel-last image with shape (Y, X, C).

    The TIFF/ND2 loads as (2, Y, X), so do NOT pass it directly as (C, Y, X).
    """
    if not use_fluor:
        return phase

    if phase.shape != fluor.shape:
        raise ValueError(
            f"Phase and fluorescence images have different shapes: {phase.shape} vs {fluor.shape}"
        )

    # Channel-last pseudo-RGB input: (Y, X, 3)
    # Channel 0: phase contrast
    # Channel 1: fluorescence
    # Channel 2: empty filler channel
    return np.stack([phase, fluor, np.zeros_like(phase)], axis=-1)


def save_label_outputs(
    segmentation: np.ndarray, phase: np.ndarray, fluor: np.ndarray, stem: str
) -> None:
    segmentation = np.asarray(segmentation)

    timestamp = int(time.time())
    Image.fromarray(segmentation.astype(np.uint32, copy=False)).save(
        f"{stem}_labels_autosave_{timestamp}.png"
    )

def get_layer_from_viewer(viewer: napari.Viewer, names: list[str]) -> np.ndarray | None:

    for name in names:
        if name in viewer.layers:
            return np.asarray(viewer.layers[name].data)

    return None

COMMITTED_OBJECTS_NAMES = ["committed_objects", "committed objects", "Committed Objects"]
AUTO_SEGMENTATION_NAMES = ["auto_segmentation", "auto segmentation", "Auto Segmentation"]

def merge_labels(series: tuple[np.ndarray]) -> np.ndarray:
    shape = None
    joint = None
    for labels in series:
        if shape is None:
            shape = labels.shape
            joint = labels.astype(np.int64).copy()
        else:
            if labels.shape != shape:
                raise ValueError(f"Layer shapes differ: {labels.shape} vs. {shape}")

        next_shifted = np.where(labels > 0, labels + joint.max(), 0)

        add_next = (joint == 0) & (next_shifted > 0)
        joint[add_next] = next_shifted[add_next]

    return joint


def get_segmentation_from_viewer(viewer: napari.Viewer) -> np.ndarray | None:
    committed = get_layer_from_viewer(viewer, COMMITTED_OBJECTS_NAMES)
    auto = get_layer_from_viewer(viewer, AUTO_SEGMENTATION_NAMES)

    if committed is not None and auto is not None:
        return merge_labels((committed, auto))
    return committed if committed is not None else auto


def patch_microsam_auto_thresholds(
    *,
    foreground_threshold: float | None = 0.60,
    labels_threshold: float | None = None,
    force_mode: str | None = None,  # use "apg" if you need labels_threshold
) -> None:
    """
    For some reason, there is no way to change these thresholds in either the micro_sam UI or through code.
    These affect how large the instance masks are
    (a higher threshold corresponds to a higher required confidence
    that the given pixel is foreground to be marked as such).
    """
    import inspect
    import numpy as np
    from micro_sam import instance_segmentation
    from micro_sam.sam_annotator import _widgets as msam_widgets
    from micro_sam.sam_annotator._state import AnnotatorState

    if force_mode is not None:
        if not hasattr(
            instance_segmentation, "_orig_get_instance_segmentation_generator"
        ):
            instance_segmentation._orig_get_instance_segmentation_generator = (
                instance_segmentation.get_instance_segmentation_generator
            )

        def get_generator(
            predictor, is_tiled, decoder=None, segmentation_mode=None, **kwargs
        ):
            if decoder is not None and segmentation_mode is None:
                segmentation_mode = force_mode
            return instance_segmentation._orig_get_instance_segmentation_generator(
                predictor,
                is_tiled=is_tiled,
                decoder=decoder,
                segmentation_mode=segmentation_mode,
                **kwargs,
            )

        instance_segmentation.get_instance_segmentation_generator = get_generator

    def instance_segmentation_impl(
        min_object_size, i=None, pbar_init=None, pbar_update=None, **kwargs
    ):
        state = AnnotatorState()
        msam_widgets._handle_amg_state(state, i, pbar_init, pbar_update)

        generate_params = inspect.signature(state.amg.generate).parameters

        if (
            foreground_threshold is not None
            and "foreground_threshold" in generate_params
        ):
            kwargs["foreground_threshold"] = foreground_threshold

        if labels_threshold is not None and "labels_threshold" in generate_params:
            kwargs["labels_threshold"] = labels_threshold

        seg = state.amg.generate(**kwargs)
        assert isinstance(seg, np.ndarray)
        return seg

    msam_widgets._instance_segmentation_impl = instance_segmentation_impl
