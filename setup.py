import platform
from setuptools import setup, Extension
import sys
import numpy


arch = platform.machine().lower()
extra_compile_args = []

if sys.platform == "win32":
    extra_compile_args += ["/O2", "/arch:AVX2", "/Qpar"]  # enables AVX/AVX2; SSE4.2 implied
elif "arm" in arch or "aarch64" in arch:
    extra_compile_args += ["-O3"]
else:
    extra_compile_args += ["-O3", "-march=x86-64", "-mavx2", "-msse4.2", "-flto", "-mfma", "-ffp-contract=fast",]

module = Extension(
    "normal_grain_merge.normal_grain_merge",
    sources=["normal_grain_merge/normal_grain_merge.c"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(
    name="normal_grain_merge",
    version="0.1.0",
    description="Normal grain merge C extension",
    ext_modules=[module],
    packages=["normal_grain_merge"],
)
