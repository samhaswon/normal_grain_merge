import numpy as np
from ngm import normal_grain_merge

base = np.zeros((100, 100, 3), dtype=np.uint8)
texture = np.zeros((100, 100, 3), dtype=np.uint8)
skin = np.zeros((100, 100, 4), dtype=np.uint8)
im_alpha = np.zeros((100, 100), dtype=np.uint8)

result = normal_grain_merge(base, texture, skin, im_alpha)
print(result.shape, result.dtype)
