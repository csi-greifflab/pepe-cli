from setuptools import setup, find_packages
import os
import sys

# Add src to the Python path to import metadata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from pepe import __version__, __package_name__, __author__, __author_email__, __description__, __homepage__, __module_name__


# Read the README file for long description
def read_readme():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        return f.read()


setup(
    name=__package_name__,
    version=__version__,
    author=__author__,
    author_email=__author_email__,
    description=__description__,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url=__homepage__,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: Linux",
        "Operating System :: macOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "sentencepiece",
        "numpy",
        "protobuf",
        "alive_progress",
    ],
    extras_require={
        "esmc": [
            "transformers @ git+https://github.com/Biohub/transformers.git@main",
            "accelerate>=0.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            f"pepe = {__module_name__}.__main__:main",
            f"{__package_name__} = {__module_name__}.__main__:main",
        ]
    },
    keywords="protein embeddings bioinformatics machine-learning nlp transformers",
    project_urls={
        "Bug Reports": f"{__homepage__}/issues",
        "Source": __homepage__,
        "Documentation": f"{__homepage__}#readme",
    },
    include_package_data=True,
)
