# normal_grain_merge

This implements a combined version of the blend modes normal and grain merge.
Grain merge is performed on *s* and *t* with the result normal-merged with *b*.
Subscripts indicate channels, with alpha (α) channels broadcast to three channels.

$$
\def\sw{0.3}
\def\b{\mathrm{b_{rgb}}}
\def\t{\mathrm{t_{rgb}}}
\def\s{\mathrm{s_{rgb}}}
\def\al{\mathrm{alpha}}
\def\ta{\mathrm{t_\alpha}}
(((\t + \s - 0.5) * \ta + \t * (1 - \ta)) * (1 - \sw) + \s * \sw) * \ta + \b * (1 - \ta)
$$

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
