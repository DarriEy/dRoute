#!/usr/bin/env python3
"""
Setup script for dMC-Route Python bindings.

Installation:
    pip install .
    
Or for development:
    pip install -e .
    
Requires:
    - CMake 3.15+
    - C++17 compiler
    - pybind11 (auto-downloaded if not found)
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Read version from single source of truth
version_file = Path(__file__).parent / "python" / "droute" / "_version.py"
version_info = {}
with open(version_file) as f:
    exec(f.read(), version_info)
__version__ = version_info["__version__"]


class CMakeExtension(Extension):
    """CMake extension for building with cmake."""
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using cmake."""
    
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DDMC_BUILD_PYTHON=ON",
            "-DDMC_BUILD_TESTS=OFF",
        ]
        
        build_args = ["--config", cfg]
        
        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Default to 4 parallel jobs
            build_args += ["--", "-j4"]
        
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        
        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True
        )
        
        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True
        )


# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text()
else:
    long_description = "Differentiable Muskingum-Cunge routing library"


setup(
    name="droute",
    version=__version__,
    author="Darri Eythorsson",
    author_email="darri.eythorsson@ucalgary.ca",
    description="Differentiable river routing library for hydrological modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/DarriEy/dRoute",
    
    ext_modules=[CMakeExtension("_droute_core")],
    cmdclass={"build_ext": CMakeBuild},
    
    packages=["droute"],
    package_dir={"": "python"},
    py_modules=["pydmc_route"],  # Backwards compatibility shim
    include_package_data=True,
    
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "torch": ["torch>=1.9.0"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    
    keywords="hydrology, routing, muskingum-cunge, differentiable, machine-learning",
    
    zip_safe=False,
)
