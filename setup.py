from distutils.core import setup

setup(
  name = 'noisifier',
  packages = ['noisifier'],
  version = '0.4.2',
  license='MIT',
  description = 'Add label noise to your dataset',
  author = 'akakream',
  author_email = 'akakream@gmail.com',
  url = 'https://github.com/akakream/noisifier',
  download_url = 'https://github.com/akakream/noisifier/archive/v0.4.2-alpha.tar.gz',
  keywords = ['NOISE', 'LABEL NOISE', 'NOISIFY', 'NOISIFIER', 'Y_TRAIN', 'CIFAR10'],
  install_requires=[            
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
