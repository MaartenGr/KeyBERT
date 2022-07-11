from setuptools import setup, find_packages

test_packages = [
    "pytest>=5.4.3",
    "pytest-cov>=2.6.1",
    "flake8>=3.6.0",
    "pre-commit>=2.2.0",
    "black>=19.3b0",
]

base_packages = [
    "sentence-transformers>=0.3.8",
    "scikit-learn>=0.22.2",
    "numpy>=1.18.5",
    "rich>=10.4.0",
]

docs_packages = [
    "mkdocs>=1.1",
    "mkdocs-material>=4.6.3",
    "mkdocstrings>=0.8.0",
]

flair_packages = ["transformers>=3.5.1", "torch>=1.4.0", "flair>=0.7"]

spacy_packages = ["spacy>=3.0.1"]

use_packages = ["tensorflow", "tensorflow_hub", "tensorflow_text"]

gensim_packages = ["gensim>=3.6.0"]

dev_packages = docs_packages + test_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="keybert",
    packages=find_packages(exclude=["notebooks", "docs"]),
    version="0.5.1",
    author="Maarten Grootendorst",
    author_email="maartengrootendorst@gmail.com",
    description="KeyBERT performs keyword extraction with state-of-the-art transformer models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MaartenGr/keyBERT",
    keywords="nlp bert keyword extraction embeddings",
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=base_packages,
    extras_require={
        "test": test_packages,
        "docs": docs_packages,
        "dev": dev_packages,
        "flair": flair_packages,
        "spacy": spacy_packages,
        "use": use_packages,
        "gensim": gensim_packages,
    },
    python_requires=">=3.6",
)
