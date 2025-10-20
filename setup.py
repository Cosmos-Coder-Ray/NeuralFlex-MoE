"""Setup script for NeuralFlex-MoE"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuraflex-moe",
    version="0.1.0",
    author="NeuralFlex Team",
    author_email="contact@neuraflex.ai",
    description="Mixture of Experts with Adaptive Reasoning - A revolutionary lightweight LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuraflex/neuraflex-moe",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.1.0",
            "pytest-cov>=5.0.0",
            "black>=24.3.0",
            "ruff>=0.3.0",
            "mypy>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuraflex-train=neuraflex_moe.scripts.train:main",
            "neuraflex-inference=neuraflex_moe.scripts.inference:main",
            "neuraflex-server=neuraflex_moe.api.server:run_server",
        ],
    },
)
