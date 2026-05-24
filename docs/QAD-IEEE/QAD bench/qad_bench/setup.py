"""Setup script for QAD-Bench v1.1."""
from setuptools import setup, find_packages
from pathlib import Path

readme = (Path(__file__).parent / "README.md").read_text(encoding="utf-8") if (Path(__file__).parent / "README.md").exists() else ""

setup(
    name="qad-bench",
    version="1.1.0",
    description="Reproducible Telecom Fraud Detection Benchmark on TeleAntiFraud-28k",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="QAD-MultiGuard Authors",
    author_email="xin.zhang@example.org",
    license="MIT",
    url="https://github.com/campus-safety/QAD-Bench",
    packages=find_packages(exclude=("tests", "tests.*", "examples", "scripts")),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scikit-learn>=1.0",
        "rouge-score>=0.1.2",
    ],
    extras_require={
        "full": [
            "datasets>=2.14",
            "librosa>=0.10",
            "torch>=2.0",
            "transformers>=4.30",
            "bert-score>=0.3",
        ],
        "gguf": ["llama-cpp-python>=0.2.50"],
    },
    entry_points={
        "console_scripts": [
            "qad-bench = qad_bench.runner:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
