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

## Usage

### Python API

```python
import pydmc_route as dmc
import numpy as np

# Create network
network = dmc.Network()

# Add reaches with topology
for i in range(10):
    reach = dmc.Reach()
    reach.id = i
    reach.length = 5000.0  # meters
    reach.slope = 0.001
    reach.manning_n = 0.035
    reach.upstream_junction_id = i
    reach.downstream_junction_id = i + 1 if i < 9 else -1  # -1 = outlet
    network.add_reach(reach)

# Add junctions for connectivity
for i in range(10):
    junc = dmc.Junction()
    junc.id = i
    junc.upstream_reach_ids = [i-1] if i > 0 else []
    junc.downstream_reach_ids = [i]
    network.add_junction(junc)

network.build_topology()

# Configure and create router
config = dmc.RouterConfig()
config.dt = 3600.0  # 1-hour timestep

router = dmc.MuskingumCungeRouter(network, config)

# Run simulation
runoff = np.random.rand(100, 10) * 0.001  # (timesteps, reaches) in m/s
outlet_Q = []

for t in range(100):
    for r in range(10):
        router.set_lateral_inflow(r, runoff[t, r])
    router.route_timestep()
    outlet_Q.append(router.get_discharge(9))  # outlet reach
```

### Fast Optimization with Enzyme

```python
import pydmc_route as dmc
import numpy as np

# Load network from topology.nc (see test_routing_with_data.py)
network = load_topology('topology.nc')

# Create Enzyme router (fast, no AD tape overhead)
router = dmc.enzyme.EnzymeRouter(network, dt=3600.0, num_substeps=4)

# Optimize Manning's n
result = dmc.enzyme.optimize(
    router,
    runoff,           # (n_timesteps, n_reaches) in m³/s
    observed,         # (n_timesteps,) observed discharge at outlet
    outlet_reach=0,   # outlet reach index
    n_epochs=30,
    lr=0.1,
    verbose=True
)

print(f"NSE: {result['nse']:.3f}")
print(f"Optimized Manning's n: {result['optimized_manning_n']}")
```

### Loading mizuRoute Topology

```python
import xarray as xr
import pydmc_route as dmc

ds = xr.open_dataset('topology.nc')
seg_ids = ds['segId'].values
down_seg_ids = ds['downSegId'].values
lengths = ds['length'].values
slopes = ds['slope'].values

# Build network
network = dmc.Network()
seg_id_to_idx = {int(sid): i for i, sid in enumerate(seg_ids)}

# Build upstream map
upstream_map = {i: [] for i in range(len(seg_ids))}
for i, down_id in enumerate(down_seg_ids):
    if int(down_id) in seg_id_to_idx:
        upstream_map[seg_id_to_idx[int(down_id)]].append(i)

# Add reaches
for i in range(len(seg_ids)):
    reach = dmc.Reach()
    reach.id = i
    reach.length = float(lengths[i])
    reach.slope = max(float(slopes[i]), 0.0001)
    reach.manning_n = 0.05
    reach.upstream_junction_id = i
    
    down_id = int(down_seg_ids[i])
    if down_id in seg_id_to_idx:
        reach.downstream_junction_id = seg_id_to_idx[down_id]
    else:
        reach.downstream_junction_id = -1  # Outlet
    
    network.add_reach(reach)

# Add junctions
for i in range(len(seg_ids)):
    junc = dmc.Junction()
    junc.id = i
    junc.upstream_reach_ids = upstream_map[i]
    junc.downstream_reach_ids = [i]
    network.add_junction(junc)

network.build_topology()
```

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

## Performance Benchmarks

Bow at Banff basin (49 reaches, 2200 km², 8760 hourly timesteps):

| Operation | Time | Throughput |
|-----------|------|------------|
| Forward pass (MC) | 2.0s | 4,500 ts/s |
| Forward pass (all 5 methods) | 18s | - |
| Optimization (30 epochs) | 31s | - |

## Citation

```bibtex
@software{dRoute2024,
  title={dRoute: Differentiable River Routing Library},
  author={Thorsson, Darri},
  year={2024},
  url={https://github.com/DarriEy/dRoute}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- CoDiPack team for the AD library
- Enzyme team for the AD compiler plugin  
- SUMMA and mizuRoute communities for inspiration
