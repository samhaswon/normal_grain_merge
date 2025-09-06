import cv2
import numpy as np

from ngm import normal_grain_merge, KernelKind
from py_ngm import apply_texture


def percent_change(new_t: float, old_t: float):
    return -(new_t - old_t) / old_t * 100


ITERATIONS = 200


base = np.zeros((100, 100, 3), dtype=np.uint8)
texture = np.zeros((100, 100, 3), dtype=np.uint8)
skin = np.zeros((100, 100, 4), dtype=np.uint8)
im_alpha = np.zeros((100, 100), dtype=np.uint8)

result_scalar = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SCALAR.value)
print(result_scalar.shape, result_scalar.dtype)

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
result_scalar = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SCALAR.value)
result_sse = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SSE42.value)
result_avx = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_AVX2.value)
result_auto = normal_grain_merge(base, texture, skin, im_alpha)
cv2.imwrite("result.png", result_scalar)
max_diff_scalar = np.abs(result_py.astype(np.int16) - result_scalar.astype(np.int16)).max()
max_diff_sse = np.abs(result_py.astype(np.int16) - result_sse.astype(np.int16)).max()
max_diff_avx = np.abs(result_py.astype(np.int16) - result_avx.astype(np.int16)).max()
max_diff_auto = np.abs(result_py.astype(np.int16) - result_auto.astype(np.int16)).max()
print("max abs diff scalar:", max_diff_scalar)
print("max abs diff sse:", max_diff_sse)
print("max abs diff avx:", max_diff_avx)
print("max abs diff auto:", max_diff_auto)
