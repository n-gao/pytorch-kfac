import setuptools
from setuptools import setup, version

with open('readme.md') as fh:
    long_description = fh.read()

setuptools.setup(
    name='torch-kfac',
    version='0.0.1',
    author='Nicholas Gao',
    author_email='nicholas.gao@tum.de',
    description='This package provides a PyTorch implementation of the KFAC optimizer by Marten et al.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/n-gao/pytorch-kfac',
    packages=setuptools.find_packages('.'),
    install_requires=['numpy'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7'
)
