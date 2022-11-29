from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='marker',
    ext_modules=[
        CUDAExtension('marker_cuda', [
            'marker_cuda.cpp',
            'marker_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
