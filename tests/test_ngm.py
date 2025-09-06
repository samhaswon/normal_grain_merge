"""
Test that the normal grain merge works as intended.
"""
import unittest

import cv2
import numpy as np

from ngm import normal_grain_merge, KernelKind
from py_ngm import apply_texture


def vertical_fill(m, n, k):
    """
    Create an m x n matrix with 0s from row 0 up to (but not including) row k,
    and 255 afterward.

    :param m: Number of rows
    :param n: Number of columns
    :param k: Cutoff row index
    :return: numpy.ndarray of shape (m, n)
    """
    mat = np.zeros((m, n), dtype=np.uint8)
    mat[k:, :] = 255
    return mat


class TestNGM(unittest.TestCase):
    """Test that the normal grain merge works as intended."""
    def setUp(self):
        """
        Load the images used.
        """
        self.base = cv2.imread("base.png")
        self.texture = cv2.imread("texture.png")
        self.skin = cv2.imread("skin.png", cv2.IMREAD_UNCHANGED)
        self.im_alpha = self.skin[..., 3]

    def test_dummy_arrays(self):
        """
        Test with some dummy arrays of 0s.
        """
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        texture = np.zeros((100, 100, 3), dtype=np.uint8)
        skin = np.zeros((100, 100, 4), dtype=np.uint8)
        im_alpha = np.zeros((100, 100), dtype=np.uint8)

        result_scalar = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SCALAR.value)
        self.assertIs(result_scalar.dtype, np.dtype(np.uint8))
        self.assertEqual(result_scalar.shape, (100, 100, 3))

    def test_basic_diff(self):
        """
        Test the common case; RGB versions of each kernel.
        """
        result_py = apply_texture(self.base, self.skin, self.texture, self.im_alpha)
        self.skin = cv2.cvtColor(
            cv2.cvtColor(
                self.skin[..., :3],
                cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        )
        # Skin is BGR at this point
        self.skin = np.dstack([self.skin, self.im_alpha])
        result_scalar = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha, KernelKind.KERNEL_SCALAR.value)
        result_sse = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha, KernelKind.KERNEL_SSE42.value)
        result_avx = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha, KernelKind.KERNEL_AVX2.value)
        result_auto = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha)
        max_diff_scalar = np.abs(result_py.astype(np.int16) - result_scalar.astype(np.int16)).max()
        max_diff_sse = np.abs(result_py.astype(np.int16) - result_sse.astype(np.int16)).max()
        max_diff_avx = np.abs(result_py.astype(np.int16) - result_avx.astype(np.int16)).max()
        max_diff_auto = np.abs(result_py.astype(np.int16) - result_auto.astype(np.int16)).max()
        self.assertLessEqual(max_diff_scalar, 1, msg=f"max abs diff scalar: {max_diff_scalar}")
        self.assertLessEqual(max_diff_sse, 1, msg=f"max abs diff sse: {max_diff_sse}")
        self.assertLessEqual(max_diff_avx, 1, msg=f"max abs diff avx: {max_diff_avx}")
        self.assertLessEqual(max_diff_auto, 1, msg=f"max abs diff auto: {max_diff_auto}")

    def test_masked_skin(self):
        """
        Test each kernel with a masked version of the skin mask.
        """
        self.skin = cv2.cvtColor(
            cv2.cvtColor(
                self.skin[..., :3],
                cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        )
        mask = vertical_fill(self.base.shape[0], self.base.shape[1], self.base.shape[1] // 2)
        new_alpha = np.bitwise_and(self.im_alpha, mask)
        self.skin = np.dstack((self.skin[..., :3], new_alpha))

        result_py = apply_texture(self.base, self.skin, self.texture, new_alpha)
        result_scalar = normal_grain_merge(self.base, self.texture, self.skin, new_alpha, KernelKind.KERNEL_SCALAR.value)
        result_sse = normal_grain_merge(self.base, self.texture, self.skin, new_alpha, KernelKind.KERNEL_SSE42.value)
        result_avx = normal_grain_merge(self.base, self.texture, self.skin, new_alpha, KernelKind.KERNEL_AVX2.value)
        result_auto = normal_grain_merge(self.base, self.texture, self.skin, new_alpha)

        max_diff_scalar = np.abs(result_py.astype(np.int16) - result_scalar.astype(np.int16)).max()
        max_diff_sse = np.abs(result_py.astype(np.int16) - result_sse.astype(np.int16)).max()
        max_diff_avx = np.abs(result_py.astype(np.int16) - result_avx.astype(np.int16)).max()
        max_diff_auto = np.abs(result_py.astype(np.int16) - result_auto.astype(np.int16)).max()
        self.assertLessEqual(max_diff_scalar, 1, msg=f"max abs diff scalar: {max_diff_scalar}")
        self.assertLessEqual(max_diff_sse, 1, msg=f"max abs diff sse: {max_diff_sse}")
        self.assertLessEqual(max_diff_avx, 1, msg=f"max abs diff avx: {max_diff_avx}")
        self.assertLessEqual(max_diff_auto, 1, msg=f"max abs diff auto: {max_diff_auto}")

    def test_masked_texture(self):
        """
        Mask off part of the texture partially to test the RGBA forms of each kernel.
        """
        mask = vertical_fill(self.base.shape[0], self.base.shape[1], self.base.shape[1] // 2)
        mask = (mask * 0.5).astype(np.uint8)
        self.texture = np.dstack((self.texture, mask))

        self.skin = cv2.cvtColor(
            cv2.cvtColor(
                self.skin[..., :3],
                cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR
        )
        result_py = apply_texture(self.base, self.skin, self.texture, self.im_alpha)
        # Skin is BGR at this point
        self.skin = np.dstack([self.skin, self.im_alpha])
        result_scalar = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha, KernelKind.KERNEL_SCALAR.value)
        result_sse = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha, KernelKind.KERNEL_SSE42.value)
        result_avx = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha, KernelKind.KERNEL_AVX2.value)
        result_auto = normal_grain_merge(self.base, self.texture, self.skin, self.im_alpha)

        max_diff_scalar = np.abs(result_py.astype(np.int16) - result_scalar.astype(np.int16)).max()
        max_diff_sse = np.abs(result_py.astype(np.int16) - result_sse.astype(np.int16)).max()
        max_diff_avx = np.abs(result_py.astype(np.int16) - result_avx.astype(np.int16)).max()
        max_diff_auto = np.abs(result_py.astype(np.int16) - result_auto.astype(np.int16)).max()
        self.assertLessEqual(max_diff_scalar, 1, msg=f"max abs diff scalar: {max_diff_scalar}")
        self.assertLessEqual(max_diff_sse, 1, msg=f"max abs diff sse: {max_diff_sse}")
        self.assertLessEqual(max_diff_avx, 1, msg=f"max abs diff avx: {max_diff_avx}")
        self.assertLessEqual(max_diff_auto, 1, msg=f"max abs diff auto: {max_diff_auto}")


if __name__ == '__main__':
    unittest.main()
