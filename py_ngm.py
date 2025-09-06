from blend_modes import grain_merge, normal
import cv2
import numpy as np


SKIN_WEIGHT = 0.3


def normal_grain_merge_py(base: np.array, texture: np.array, skin: np.array, im_alpha: np.array):
    """
    Python Implementation of fused normal and grain merge.
    :param base: RGB or RGBA image base.
    :param texture: RGB or RGBA texture.
    :param skin: RGBA skin layer.
    :param im_alpha: Skin alpha channel.
    :return: RGB np.ndarray.
    """
    base = base / 255.0
    texture = texture / 255.0
    skin = skin / 255.0
    im_alpha = im_alpha / 255.0

    if base.shape[2] == 4:
        base = base[..., :3]

    if texture.shape[2] == 4:
        skin = np.dstack([skin, im_alpha])
        texture_alpha = texture[..., 3] * im_alpha
        texture_alpha = np.dstack([texture_alpha, texture_alpha, texture_alpha, texture_alpha])
        base = np.dstack([base, np.ones(base.shape[:2])])
    else:
        texture_alpha = np.dstack([im_alpha, im_alpha, im_alpha])
    inverse_tpa = 1 - texture_alpha

    # Grain merge
    gm_out = (
        (
            np.clip(texture + skin - 0.5, 0.0, 1.0) *
            texture_alpha + texture * inverse_tpa
        ) *
        (1 - SKIN_WEIGHT) + (skin * SKIN_WEIGHT)
    )
    np.nan_to_num(gm_out, copy=False)

    # Normal merge
    n_out = gm_out * texture_alpha + base * inverse_tpa
    if n_out.shape[2] == 4:
        return np.uint8(n_out[..., :3] * 255.0)
    return np.uint8(n_out * 255.0)


def apply_texture(original: np.ndarray, skin: np.array, texture: np.array, im_alpha: np.array) -> np.array:
    """
    Python Implementation of fused normal and grain merge with preprocessing.
    :param original: RGB or RGBA image base.
    :param texture: RGB or RGBA texture.
    :param skin: RGBA skin layer.
    :param im_alpha: Skin alpha channel.
    :return: RGB np.ndarray.
    """
    bw_masked = cv2.cvtColor(
        cv2.cvtColor(
            skin,
            cv2.COLOR_BGRA2GRAY),
        cv2.COLOR_GRAY2BGR)

    mix_layer = normal_grain_merge_py(
        original.astype(np.float32),
        texture.astype(np.float32),
        bw_masked.astype(np.float32),
        im_alpha.astype(np.float32)
    )
    return mix_layer


def normal_grain_merge_old(original: np.ndarray, texture: np.array, skin: np.array, im_alpha: np.array) -> np.array:
    """
    Python Implementation of non-fused normal and grain merge.
    This is slow.
    :param original: RGB or RGBA image base.
    :param texture: RGB or RGBA texture.
    :param skin: RGBA skin layer.
    :param im_alpha: Skin alpha channel.
    :return: RGB np.ndarray.
    """
    bw_masked = cv2.cvtColor(skin, cv2.COLOR_BGR2BGRA).astype(np.float32)
    texture_float = np.dstack((texture, im_alpha)).astype(np.float32)
    mix_layer = grain_merge(texture_float, bw_masked, 1.0)
    mix_layer = np.uint8(
        normal(
            cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGRA).astype(np.float32),
            mix_layer,
            1.0)
    )
    return mix_layer
