import cv2
import numpy as np

from ngm import normal_grain_merge, KernelKind


SKIN_WEIGHT = 0.3


def normal_grain_merge_py(base: np.array, texture: np.array, skin: np.array, im_alpha: np.array):
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


base = np.zeros((100, 100, 3), dtype=np.uint8)
texture = np.zeros((100, 100, 3), dtype=np.uint8)
skin = np.zeros((100, 100, 4), dtype=np.uint8)
im_alpha = np.zeros((100, 100), dtype=np.uint8)

result = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SCALAR.value)
print(result.shape, result.dtype)

# actual test
base = cv2.imread("base.png")
texture = cv2.imread("texture.png")
skin = cv2.imread("skin.png", cv2.IMREAD_UNCHANGED)
im_alpha = skin[..., 3]
result_py = apply_texture(base, skin, texture, im_alpha)
skin = cv2.cvtColor(
        cv2.cvtColor(
            skin[..., :3],
            cv2.COLOR_BGR2GRAY),
        cv2.COLOR_GRAY2BGR
    )
# Skin is BGR at this point
skin = np.dstack([skin, im_alpha])
result = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SCALAR.value)
cv2.imwrite("result.png", result)
print("max abs diff:", np.abs(result_py.astype(np.int16) - result.astype(np.int16)).max())

