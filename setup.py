from setuptools import setup, find_packages

setup(
    name='mlae',
    description='Maximum Likelihood Autoencoder',
    version='0.1dev',
    python_requires='>=3.6, <=3.10',
    packages=find_packages(exclude=['tests']),
    license='MIT',
    author='Felix Draxler, Peter Sorrenson',
    author_email='felix.draxler@iwr.uni-heidelberg.de, peter.sorrenson@iwr.uni-heidelberg.de',
    url='https://github.com/vislearn/MLAE'
)
