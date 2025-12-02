# dRoute: Differentiable River Routing Library

A high-performance differentiable routing library for hydrological modeling. Implements multiple routing methods with automatic differentiation support for parameter optimization.

## Quick Start

### 1. Build

```bash
git clone https://github.com/DarriEy/dRoute.git
cd dRoute
mkdir build && cd build
cmake .. -DDMC_BUILD_PYTHON=ON
make -j4
```

### 2. Test Installation

```bash
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python
python -c "import pydmc_route as dmc; print(f'dRoute v{dmc.__version__} loaded')"
```

### 3. Run with Sample Data

```bash
# Forward pass (all routing methods)
python python/test_routing_with_data.py --data-dir data

# Parameter optimization with fast Enzyme kernels
python python/test_routing_with_data.py --data-dir data --optimize --fast --methods mc
```

## Features

- **5 routing methods**: Muskingum-Cunge, KWT-Soft, Diffusive Wave, IRF, Lag
- **Automatic differentiation**: CoDiPack (tape-based) and Enzyme (source-to-source) backends
- **Network topology**: Full support for river networks with tributaries and confluences
- **mizuRoute compatible**: Reads standard `topology.nc` files
- **Fast optimization**: Gradient-based calibration of Manning's n

## Routing Methods

| Method | Class | Gradients | Speed | Use Case |
|--------|-------|-----------|-------|----------|
| Muskingum-Cunge | `MuskingumCungeRouter` | Full AD | ~4,500/s | Production routing |
| KWT-Soft | `SoftGatedKWT` | Full AD | ~4,000/s | Differentiable wave tracking |
| Diffusive Wave | `DiffusiveWaveIFT` | IFT | ~3,000/s | Physics-based routing |
| IRF | `IRFRouter` | Full AD | ~1,100/s | Unit hydrograph |
| Lag | `LagRouter` | Analytical | ~22,000/s | Simple delay |

## Installation

### Prerequisites

- C++17 compiler (GCC 9+, Clang 10+, Apple Clang 12+)
- CMake 3.15+
- Python 3.8+ with NumPy

### macOS (Apple Silicon)

```bash
# Install dependencies
brew install cmake

# Build
git clone https://github.com/DarriEy/dRoute.git
cd dRoute
mkdir build && cd build
cmake .. -DDMC_BUILD_PYTHON=ON
make -j$(sysctl -n hw.ncpu)

# Set up Python path
echo 'export PYTHONPATH=$PYTHONPATH:'$(pwd)'/python' >> ~/.zshrc
source ~/.zshrc
```

### Linux (HPC)

```bash
module load cmake gcc python

git clone https://github.com/DarriEy/dRoute.git
cd dRoute
mkdir build && cd build
cmake .. -DDMC_BUILD_PYTHON=ON -DCMAKE_CXX_COMPILER=g++
make -j8

export PYTHONPATH=$PYTHONPATH:$(pwd)/python
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `DMC_BUILD_PYTHON` | OFF | Build Python bindings |
| `DMC_ENABLE_NETCDF` | OFF | Enable NetCDF topology I/O |
| `DMC_ENABLE_ENZYME` | OFF | Enable Enzyme AD backend |
| `DMC_ENABLE_OPENMP` | OFF | Enable OpenMP parallelization |
| `DMC_BUILD_TESTS` | ON | Build C++ test suite |


## Data Directory Structure

```
data/
├── settings/
│   └── dRoute/
│       └── topology.nc          # River network topology
├── simulations/
│   └── SUMMA/
│       └── run_1_timestep.nc    # SUMMA runoff output
└── observations/
    └── streamflow/
        └── streamflow.csv       # Observed discharge
```

### topology.nc Variables

| Variable | Dimensions | Description |
|----------|------------|-------------|
| `segId` | (seg) | Segment IDs |
| `downSegId` | (seg) | Downstream segment ID |
| `length` | (seg) | Reach length (m) |
| `slope` | (seg) | Channel slope (m/m) |
| `hruId` | (hru) | HRU IDs |
| `hruToSegId` | (hru) | HRU to segment mapping |
| `area` | (hru) | HRU area (m²) |

## Testing

```bash
# C++ tests
cd build
ctest --output-on-failure

# Python tests
cd ..
PYTHONPATH=build/python python python/test_routing_with_data.py --data-dir data

# Quick synthetic test (no data needed)
PYTHONPATH=build/python python python/test_routing_with_data.py
```

## Learnable Parameters

All routers support gradients for:
- `manning_n`: Manning's roughness coefficient

Muskingum-Cunge additionally supports:
- `width_coef`, `width_exp`: W = a × Q^b
- `depth_coef`, `depth_exp`: D = c × Q^d
