# normal_grain_merge

This implements a combined version of the blend modes normal and grain merge.
Grain merge is performed on *s* and *t* with the result normal-merged with *b*.
Subscripts indicate channels, with alpha (α) channels broadcast to three channels.

$$
(((\mathrm{t_{rgb}} + \mathrm{s_{rgb}} - 0.5) * \mathrm{t_\alpha} + \mathrm{t_{rgb}} * (1 - \mathrm{t_\alpha})) * (1 - 0.3) + \mathrm{s_{rgb}} * 0.3) * \mathrm{t_\alpha} + \mathrm{b_{rgb}} * (1 - \mathrm{t_\alpha})
$$

## Installation

```shell
pip install normal-grain-merge
```

## Usage
```py
import numpy as np
from normal_grain_merge import normal_grain_merge, KernelKind


# Example arrays
base = np.zeros((100, 100, 3), dtype=np.uint8)
texture = np.zeros((100, 100, 3), dtype=np.uint8)
skin = np.zeros((100, 100, 4), dtype=np.uint8)
im_alpha = np.zeros((100, 100), dtype=np.uint8)

result_scalar = normal_grain_merge(base, texture, skin, im_alpha, KernelKind.KERNEL_SCALAR.value)
print(result_scalar.shape, result_scalar.dtype)
```

There are three kernels implemented in this module as defined in `KernelKind`.

- `KERNEL_AUTO`: Automatically chooses the kernel, preferring AVX2
- `KERNEL_SCALAR`: Portable scalar implementation.
- `KERNEL_SSE42`: SSE4.2 intrinsics kernel. Likely better on AMD CPUs.
- `KERNEL_AVX2`: AVX2 intrinsics kernel. Likely better on Intel CPUs.

### Parameters

All input matrices should have the same height and width.

#### `base`

RGB or RGBA, dropping the alpha channel if it exists.
The base image for application.

#### `texture`

RGB or RGBA, applying the alpha if it exists.
This is the texture to be applied.

#### `skin`

RGBA, the segmented portion of base to texture.
The "skin" of the object the texture is to be applied to.

#### `im_alpha`

The alpha of parameter `skin`. 
This is mostly a holdover from the Python implementation to deal with NumPy.

#### `kernel`

One of `KernelKind`.

## Performance

The entire reason for me writing this was NumPy being slow when this operation is in the hot path.
So, I decided to write a SIMD version that does the type casting outside NumPy with only the intermediate values being in FP32.

How much of a speedup is this? All numbers are from a Ryzen 7 4800H running Ubuntu 24.04 and Python 3.12.3.

| Method/Kernel     | Average Iteration Time |
|-------------------|------------------------|
| C scalar kernel   | 0.015599s              |
| C SSE4.2 kernel   | 0.007122s              |
| C AVX2 kernel     | 0.011721s              |
| NumPy version     | 0.150046s              |
| Old NumPy version | 0.235400s              |

| Method Comparison  | Speedup  |
|--------------------|----------|
| NumPy -> scalar    | 89.6042% |
| NumPy -> SSE4.2    | 95.2537% |
| NumPy -> AVX2      | 92.1881% |
| Old np -> SSE4.2   | 96.9747% |
| Old np -> AVX2     | 95.0206% |
| C scalar -> SSE4.2 | 54.3442% |
| C scalar -> AVX2   | 24.8552% |
