import platform
import sys
import pkg_resources
from setuptools import setup, find_packages, Extension
import versioneer
from Cython.Build import cythonize
from glob import glob


SYS_PLATFORM = platform.system().lower()
IS_LINUX = 'linux' in SYS_PLATFORM
IS_OSX = 'darwin' == SYS_PLATFORM
IS_WIN = 'windows' == SYS_PLATFORM


# ---- C/C++ EXTENSIONS ---- #
cython_modules = [
    'menpo/external/skimage/_warps_cy.pyx',
    'menpo/feature/windowiterator.pyx',
    'menpo/feature/gradient.pyx',
    'menpo/image/patches.pyx',
    'menpo/shape/mesh/normals.pyx'
]


cython_exts = cythonize(cython_modules)
cython_exts += cythonize([
        Extension('menpo.transform.piecewiseaffine.fastpwa',
                  include_dirs=['menpo/transform/piecewiseaffine/fastpwa/opcode',
                                'menpo/transform/piecewiseaffine/fastpwa/opcode/Ice'],
                  language='c++',
                  sources=['menpo/transform/piecewiseaffine/fastpwa.pyx'] +
                          list(glob('menpo/transform/piecewiseaffine/fastpwa/opcode/OPC_*.cpp')) +
                          list(glob('menpo/transform/piecewiseaffine/fastpwa/opcode/Ice/Ice*.cpp')) +
                          ['menpo/transform/piecewiseaffine/fastpwa/opcode/Opcode.cpp',
                           'menpo/transform/piecewiseaffine/fastpwa/opcode_wrapper.cpp'])
])
# Perform a small amount of gymnastics to improve the compilation output on
# each platform (including finding numpy without importing it)
numpy_incl = pkg_resources.resource_filename('numpy', 'core/include')
for ext in cython_exts:
    if numpy_incl not in ext.include_dirs:
        ext.include_dirs.append(numpy_incl)
    if IS_LINUX or IS_OSX:
        ext.extra_compile_args.append('-Wno-unused-function')


# Please see conda/meta.yaml for other binary dependencies
install_requires = ['numpy>=1.10,<2.0',
                    'scipy>=0.16,<1.0',
                    'matplotlib>=1.4,<2.0',
                    'pillow>=3.0,<4.0',
                    'cython>=0.23']

if sys.version_info.major == 2:
    install_requires.append('pathlib==1.0')

setup(name='menpo',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description='A Python toolkit for handling annotated data',
      author='The Menpo Team',
      author_email='hello@menpo.org',
      ext_modules=cython_exts,
      packages=find_packages(),
      install_requires=install_requires,
      package_data={'menpo': ['data/*',
                              'feature/cpp/*.cpp',
                              'feature/cpp/*.h',
                              'transform/piecewiseaffine/fastpwa/*.c',
                              'transform/piecewiseaffine/fastpwa/*.h'],
                    '': ['*.pxd', '*.pyx']},
      tests_require=['nose', 'mock']
)
