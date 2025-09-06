from setuptools import setup, Extension
import numpy

module = Extension(
    "ngm.normal_grain_merge",
    sources=["ngm/normal_grain_merge.c"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="ngm",
    version="0.0.1",
    description="Normal grain merge C extension",
    ext_modules=[module],
    packages=["ngm"],
)
