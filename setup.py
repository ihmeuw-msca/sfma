from setuptools import setup

setup(name='sfma',
      version='0.0.0',
      description='stochastic frontier meta-analysis tool',
      url='https://github.com/ihmeuw-msca/SFMA',
      author='Marlena Bannick, Peng Zheng',
      author_email='mnorwood@uw.edu, zhengp@uw.edu',
      license='MIT',
      packages=['sfma'],
      package_dir={'': 'src'},
      install_requires=['numpy',
                        'scipy',
                        'pandas',
                        'pytest',
                        'ipopt',
                        'limetr',
                        'xspline'],
      zip_safe=False)
