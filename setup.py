from setuptools import setup
from setuptools import find_packages

long_description = '''
DeepFryer is a framework for the analysis of RNA-seq data using Deep Learning in Python. It integrates a lot of functionalities that 
are useful for this sort of analysis.

DeepFryer is compatible with Python 3.x
and is distributed under the MIT liense.
'''

setup(name='DeepFryer',
      version='0.1.0',
      description='Deep Learning for RNA-seq data',
      long_description=long_description,
      author='Marcos Camara-Donoso',
      author_email='marcos.camara@crg.eu',
      url='https://github.com/guigolab/DeepFryer',
      download_url='https://github.com/guigolab/DeepFryer/tarball/0.1.0',
      license='MIT',
      install_requires=['pandas',
                        'matplotlib',
                        'scikit-learn',
                        'pyarrow', 
                        'numpy',
                        'scipy',
                        'six',
                        'pyyaml',
                        'h5py',
                        'rpy2',
                        ],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov',
                    'pandas',
                    'requests'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
