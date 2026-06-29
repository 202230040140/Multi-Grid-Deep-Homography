from __future__ import annotations

import sys
import types


LEGACY_MODULES = (
    "models",
    "H_model",
    "tensorDLT",
    "tensorDLT_local",
    "tf_spatial_transform",
    "tf_spatial_transform_local",
    "utils",
)


def purge_legacy_modules() -> None:
    for name in LEGACY_MODULES:
        sys.modules.pop(name, None)


def install_tf1_compat():
    """Install enough TensorFlow 1.x surface area for the original MGDH code.

    The upstream repo imports ``tensorflow.contrib`` and uses TF1 graph/session
    APIs. Current TensorFlow keeps most of that under ``tensorflow.compat.v1``,
    while ``tf.contrib.slim`` lives in the separately installed ``tf-slim``.
    """
    import tensorflow.compat.v1 as tf
    import tf_slim as slim

    tf.disable_v2_behavior()

    if not hasattr(tf.image, "resize_images"):
        def resize_images(images, size, method=0, align_corners=False, preserve_aspect_ratio=False, name=None):
            resize_method = tf.image.ResizeMethod.BILINEAR if method == 0 else method
            return tf.image.resize(
                images,
                size,
                method=resize_method,
                preserve_aspect_ratio=preserve_aspect_ratio,
                name=name,
            )

        tf.image.resize_images = resize_images
    if not hasattr(tf, "extract_image_patches"):
        tf.extract_image_patches = tf.image.extract_patches
    if not hasattr(tf, "matrix_solve"):
        tf.matrix_solve = tf.linalg.solve

    contrib = types.ModuleType("tensorflow.contrib")
    layers = types.ModuleType("tensorflow.contrib.layers")
    layers.conv2d = slim.conv2d
    contrib.slim = slim
    contrib.layers = layers
    tf.contrib = contrib

    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.contrib"] = contrib
    sys.modules["tensorflow.contrib.slim"] = slim
    sys.modules["tensorflow.contrib.layers"] = layers
    return tf
