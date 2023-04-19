from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Simple synthetic dataset of shapes with known ground truth concept -> class associations"

setup(
    name="elements",
    version=VERSION,
    author="AngusNicolson",
    author_email="<angusjnicolson@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
)
