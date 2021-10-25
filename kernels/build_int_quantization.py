from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='int_quantization',
      ext_modules=[CUDAExtension(name='int_quantization',
                                 sources=['int_quantization.cpp', 'gemmlowp.cu'],
                                 extra_compile_args={'cxx': ['-g']})
                   ],
      cmdclass={'build_ext': BuildExtension})


# for installation execute:
# > python build_int_quantization.py install
# record list of all installed files:
# > python build_int_quantization.py install --record files.txt
