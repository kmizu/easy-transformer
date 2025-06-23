"""Setup configuration for easy-transformer package."""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Core requirements
install_requires = read_requirements("requirements.txt")

# Optional requirements
extras_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
        "isort>=5.10.0",
        "pre-commit>=2.19.0",
    ],
    "docs": read_requirements("requirements-mkdocs.txt"),
    "notebooks": [
        "jupyter>=1.0.0",
        "ipywidgets>=7.7.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    "cuda": [
        "nvidia-ml-py3>=7.352.0",
    ],
}

# All optional requirements
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="easy-transformer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive tutorial on Transformers for programming language implementers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/easy-transformer",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/easy-transformer/issues",
        "Documentation": "https://yourusername.github.io/easy-transformer",
        "Source Code": "https://github.com/yourusername/easy-transformer",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "easy-transformer=easy_transformer.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "easy_transformer": ["data/*.json", "configs/*.yaml"],
    },
)