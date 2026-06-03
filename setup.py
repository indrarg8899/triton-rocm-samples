from setuptools import setup, find_packages

setup(
    name="triton-rocm-samples",
    version="1.0.0",
    description="Triton kernel samples optimized for AMD ROCm GPUs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="indrarg8899",
    url="https://github.com/indrarg8899/triton-rocm-samples",
    license="MIT",
    packages=find_packages(exclude=["tests*", "benchmarks*", "docs*"]),
    python_requires=">=3.10",
    install_requires=[
        "triton>=2.1.0",
        "torch>=2.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-timeout>=2.1.0",
            "matplotlib>=3.7.0",
            "numpy>=1.24.0",
            "pyyaml>=6.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    keywords="triton rocm amd gpu kernel matrix-multiplication flash-attention softmax",
)
