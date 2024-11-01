# setup.py
from setuptools import setup, find_packages

setup(
    name="cgp_cnn",
    version="0.1.0",
    author="Rogfel Thompson Martínez",
    author_email="rogfel@gmail.com",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
        'torch',
        'scikit-learn'
    ]
)