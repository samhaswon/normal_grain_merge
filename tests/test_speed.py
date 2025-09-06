"""
Performance testing for each kernel and Python baseline.
"""
import time
import unittest

import cv2
import numpy as np

from ngm import normal_grain_merge, KernelKind
from py_ngm import normal_grain_merge_py, normal_grain_merge_old


def percent_change(new_t: float, old_t: float):
    """
    Calculates the (inverted) percent change between two numbers.
    :param new_t: The new value.
    :param old_t: The old value.
    :return: The percent change. Positive is speedup.
    """
    return -(new_t - old_t) / old_t * 100


class TestNGM(unittest.TestCase):
    """Performance testing for each kernel and Python baseline."""
    def test_perf(self):
        """Perf test function"""
        # Number of iterations for each test*
        ITERATIONS = 200
        global_start = time.perf_counter()
        base = cv2.imread("base.png")
        texture = cv2.imread("texture.png")
        skin = cv2.imread("skin.png", cv2.IMREAD_UNCHANGED)
        im_alpha = skin[..., 3]
        skin = cv2.cvtColor(
            cv2.cvtColor(
                skin[..., :3],
                cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        )
        # Skin is BGR at this point
        skin = np.dstack([skin, im_alpha])

        # Scaler kernel
        start_c_scalar = time.perf_counter()
        for _ in range(ITERATIONS):
            result = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SCALAR.value)
        end_c_scalar = time.perf_counter()

        # SSE4.2 kernel
        start_c_sse = time.perf_counter()
        for _ in range(ITERATIONS):
            result = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SSE42.value)
        end_c_sse = time.perf_counter()

        # AVX2 kernel
        start_c_avx = time.perf_counter()
        for _ in range(ITERATIONS):
            result = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_AVX2.value)
        end_c_avx = time.perf_counter()

        # NumPy "just do less" version.
        skin = skin[..., :3]
        start_py = time.perf_counter()
        for _ in range(ITERATIONS):
            result = normal_grain_merge_py(base, texture, skin, im_alpha)
        end_py = time.perf_counter()

        # * Except this test, it's slow.
        # Naive approach.
        start_py_old = time.perf_counter()
        for _ in range(ITERATIONS // 2):
            result = normal_grain_merge_old(base, texture, skin, im_alpha)
        end_py_old = time.perf_counter()

        # Calculate the average time per iteration
        c_avg_scalar = (end_c_scalar - start_c_scalar) / ITERATIONS
        c_avg_sse = (end_c_sse - start_c_sse) / ITERATIONS
        c_avg_avx = (end_c_avx - start_c_avx) / ITERATIONS
        np_avg = (end_py - start_py) / ITERATIONS
        np_old_avg = (end_py_old - start_py_old) / (ITERATIONS / 2)
        end = time.perf_counter()

        print(f"C scalar kernel: {c_avg_scalar:.6f}s\n"
              f"C SSE4.2 kernel: {c_avg_sse:.6f}s\n"
              f"C AVX2 kernel:   {c_avg_avx:.6f}s\n"
              f"NumPy version:   {np_avg:.6f}s\n"
              f"Old np version:  {np_old_avg:.6f}s\n"
              f"NumPy -> scalar: {percent_change(c_avg_scalar, np_avg):.4f}%\n"
              f"NumPy -> SSE4.2: {percent_change(c_avg_sse, np_avg):.4f}%\n"
              f"NumPy -> AVX2:   {percent_change(c_avg_avx, np_avg):.4f}%\n"
              f"Old np -> SSE:   {percent_change(c_avg_sse, np_old_avg):.4f}%\n"
              f"C scalar -> SSE: {percent_change(c_avg_sse, c_avg_scalar):.4f}%\n"
              f"C scalar -> AVX: {percent_change(c_avg_avx, c_avg_scalar):.4f}%\n")
        print(f"Test time: {end - global_start:.4f}s")


if __name__ == '__main__':
    unittest.main()
