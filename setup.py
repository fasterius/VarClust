from setuptools import setup

setup(name='clustervariants',
      version='0.1.0',
      description='Cluster single nucleotide variants from HTS data',
      author='Erik Fasterius',
      author_email='erikfas@kth.se',
      url='http://github.com/fasterius/variantclustering',
      license='MIT',
      packages=['clustervariants'],
      scripts=[
          'bin/create_distance_matrix',
           'bin/cluster_heatmap',
           'bin/cluster_tsne'
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
