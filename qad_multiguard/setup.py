"""QAD-MultiGuard package setup."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="qad-multiguard",
    version="1.0.0",
    description="Reference implementation of QAD-MultiGuard "
                "(quantization-aware distillation for telecom fraud detection)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="QAD-MultiGuard contributors",
    license="MIT",
    packages=find_packages(exclude=["tests", "scripts"]),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
        "figures": ["matplotlib>=3.5"],
        "reproduce": ["torch>=2.0"],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
