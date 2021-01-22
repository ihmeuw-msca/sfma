from setuptools import find_packages
from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.misc_util import get_info


def configuration(parent_package='', top_path=''):
    info = get_info('npymath')

    config = Configuration('sfa_utils',
                            parent_package,
                            top_path)
    config.add_extension('npufunc',
                            ['src/sfma/log_erfc.c'],
                            extra_info=info)

    return config


setup(name='sfma',
      version='0.0.0',
      description='stochastic frontier meta-analysis tool',
      url='https://github.com/ihmeuw-msca/SFMA',
      author='Marlena Bannick, Peng Zheng',
      author_email='mnorwood@uw.edu, zhengp@uw.edu',
      license='MIT',
      package_dir={'': 'src'},
      packages=find_packages(where='src'),
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pytest',
                        'ipopt',
                        'limetr',
                        'xspline',
                        'anml'],
      zip_safe=False,
      configuration=configuration)
