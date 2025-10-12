from enum import Enum


class KernelKind(Enum):
    KERNEL_AUTO = "auto"
    KERNEL_SCALAR = "scalar"
    KERNEL_SSE42 = "sse42"
    KERNEL_AVX2 = "avx2"
