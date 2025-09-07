# ngm/normal_grain_merge.pyi
from typing import Literal
import numpy as np
from .kernel_kind import KernelKind

def normal_grain_merge(
    base: np.ndarray,
    texture: np.ndarray,
    skin: np.ndarray,
    im_alpha: np.ndarray,
    kernel: Literal["auto", "scalar", "sse42", "avx2"] | KernelKind = "auto",
) -> np.ndarray:
    """
    Performs a combined merge: grain merge of skin and texture,
    then a normal merge of that result on base.
    Channel ordering doesn't matter as long as it is consistent.
    :param base: The base RGB image.
    :param texture: The texture, either RGB or RGBA.
    :param skin: The RGBA skin cutout.
    :param im_alpha: The alpha from the cutout.
    :param kernel: Which kernel to use.
    The `auto` kernel chooses between avx2 and sse4.2 when compiled with gcc and uses `scaler` on Windows.
    The `scalar` kernel is a portable implementation that relies on the compiler for SIMD.
    The `sse42` kernel uses SSE4.2 intrinsics.
    The `avx2` kernel uses AVX2 intrinsics.
    :return: RGB np.ndarray.
    """
    ...
