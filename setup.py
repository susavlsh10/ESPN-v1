from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GPUDirect',
    ext_modules=[
        CUDAExtension('GPUDirect', [
            'GPUDirect.cu', 
        ],
        extra_link_args=['-L/home/grads/s/sls7161/anaconda3/lib/', '-lcufile', '-lcuda'],
        include_dirs = ['/home/grads/s/sls7161/anaconda3/include/', '/home/grads/s/sls7161/Documents/IR_kernels/ESPN/gds_python/']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
