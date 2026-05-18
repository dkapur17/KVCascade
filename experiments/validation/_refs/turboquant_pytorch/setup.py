from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='turboquant-pytorch',
    version='0.1.2',
    packages=['turboquant'],
    ext_modules=[
        CppExtension(
            name='turboquant.core',  
            sources=['turboquant/core.cpp']
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)