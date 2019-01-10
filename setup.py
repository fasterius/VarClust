from setuptools import setup

setup(name='VarClust',
      version='0.2.1',
      description='Variant clustering of high-throughput sequencing data',
      author='Erik Fasterius',
      author_email='erik.fasterius@outlook.com',
      url='http://github.com/fasterius/varclust',
      license='MIT',
      packages=['varclust'],
      scripts=[
          'bin/varclust_create_profiles',
          'bin/varclust_distance_matrix',
          'bin/varclust_heatmap',
          'bin/varclust_pseudo',
          'bin/varclust_tsne'
      ],
      install_requires=[
          'argparse',
          'pandas',
          'pyvcf',
          'numpy',
          'matplotlib',
          'sklearn',
          'seaborn',
          'scipy'
      ],
      zip_safe=False)
