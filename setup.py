from setuptools import setup, find_packages
 
classifiers = [
  # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
  'Development Status :: 3 - Alpha',
  'Intended Audience :: Developers',
  'Operating System :: Microsoft :: Windows :: Windows 10',
  'Topic :: Scientific/Engineering :: Machine Learning',
  'Topic :: Scientific/Engineering :: Visualization',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3'
]
 
setup(
  name='mlpipeline_analyzer',
  version='0.0.1',
  description='Python package that analyze, visualize and suggest any changes to the machine learning pipeline ',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',  
  author='Tharun Kumar Reddy Karasani',
  author_email='karasani.tarunreddy@gmail.com',
  url = 'https://github.com/TharunKumarReddy5/ml-pipeline-analyzer',
  download_url = '',
  license='MIT', 
  classifiers=classifiers,
  keywords=['machinelearning', 'visualizer', 'analyzer', 'mlpipeline', 'suggestion'],
  packages=find_packages(),
  install_requires=[''] 
)