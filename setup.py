import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the Anaconda include and lib directories
anaconda_include_dir = os.path.join(os.path.expanduser('~'), 'anaconda3', 'include')
anaconda_lib_dir = os.path.join(os.path.expanduser('~'), 'anaconda3', 'lib')

setup(
    name='GPUDirect',
    ext_modules=[
        CUDAExtension('GPUDirect', [
            'GPUDirect.cu', 
        ],
        extra_link_args=[f'-L{anaconda_lib_dir}', '-lcufile', '-lcuda'],
        include_dirs=[anaconda_include_dir, current_dir]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })