from setuptools import setup, find_packages

classifiers = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    #'Topic :: Scientific/Engineering :: Machine Learning',
    #'Topic :: Scientific/Engineering :: Visualization',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8'
]

setup(
    name='mlpipeline_analyzer',
    version='0.0.2',
    description='Python package that analyze, visualize and suggest any changes to the machine learning pipeline ',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read() + '\n\n' + open('CHANGELOG.txt').read(),
    author='Tharun Kumar Reddy Karasani',
    author_email='karasani.tarunreddy@gmail.com',
    url='https://github.com/TharunKumarReddy5/ml-pipeline-analyzer',
    download_url='',
    license='MIT',
    classifiers=classifiers,
    keywords=['machinelearning', 'visualizer', 'analyzer', 'mlpipeline', 'suggestion'],
    packages=find_packages(),
    install_requires=['']
)
