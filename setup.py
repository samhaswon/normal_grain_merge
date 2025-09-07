from setuptools import setup, Extension
import numpy
import sys

extra_compile_args = []

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args += ["/O2", "/arch:AVX2", "/Qpar"]  # enables AVX/AVX2; SSE4.2 implied
else:
    extra_compile_args += ["-O3", "-march=x86-64-v3", "-mavx2", "-msse4.2"]

module = Extension(
    "normal_grain_merge.normal_grain_merge",
    sources=["normal_grain_merge/normal_grain_merge.c"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
)

setup(
    name="normal_grain_merge",
    version="0.0.1",
    description="Normal grain merge C extension",
    ext_modules=[module],
    packages=["normal_grain_merge"],
)
