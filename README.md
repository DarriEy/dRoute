# dRoute: Differentiable River Routing Library

A high-performance differentiable routing library for hydrological modeling, supporting automatic differentiation via dual AD backends (CoDiPack and Enzyme). Designed for integration with land surface models like SUMMA and routing frameworks like mizuRoute.

## Features

- **Multiple routing methods**: Muskingum-Cunge, Kinematic Wave (KWT), IRF, Lag, Diffusive Wave
- **Dual AD backends**: CoDiPack (tape-based) and Enzyme (source-to-source)
- **Network topology**: Full support for river network connectivity with junctions/confluences
- **Python bindings**: Complete API via pybind11
- **mizuRoute compatible**: Reads standard topology.nc files

## Routing Methods

| Method | Class | Gradients | Speed | Description |
|--------|-------|-----------|-------|-------------|
| Muskingum-Cunge | `MuskingumCungeRouter` | Full AD (5 params) | Fast | Industry-standard flood routing |
| KWT-Soft | `SoftGatedKWT` | Full AD | Medium | Differentiable Lagrangian wave tracking |
| Diffusive-IFT | `DiffusiveWaveIFT` | Implicit Function Theorem | Medium | Physics-based with exact gradients |
| IRF | `IRFRouter` | Full AD (soft-masked) | Medium | Unit hydrograph convolution |
| Lag | `LagRouter` | Analytical (weak) | Fast | Simple time delay |

## Installation

### Prerequisites

- C++17 compiler (GCC 9+, Clang 10+, Apple Clang 12+)
- CMake 3.15+
- Python 3.8+ (for bindings)

### Build from Source

```bash
git clone https://github.com/DarriEy/dRoute.git
cd dRoute
mkdir build && cd build

# Basic build
cmake .. -DDMC_BUILD_PYTHON=ON
make -j4

# With Enzyme AD (requires ClangEnzyme plugin)
cmake .. -DDMC_BUILD_PYTHON=ON \
         -DDMC_ENABLE_ENZYME=ON \
         -DENZYME_PLUGIN=/path/to/ClangEnzyme-19.dylib
make -j4

# Add Python module to path
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

## Quick Start

### Python API

```python
import pydmc_route as dmc
import numpy as np

# Create network
network = dmc.Network()

# Add reaches
for i in range(10):
    reach = dmc.Reach()
    reach.id = i
    reach.length = 5000.0  # meters
    reach.slope = 0.001
    reach.manning_n = 0.035
    reach.geometry.width_coef = 7.2
    reach.geometry.width_exp = 0.5
    network.add_reach(reach)

network.build_topology()

# Configure router
config = dmc.RouterConfig()
config.dt = 3600.0  # 1-hour timestep
config.enable_gradients = True

# Create router (choose method)
router = dmc.MuskingumCungeRouter(network, config)
# or: router = dmc.SoftGatedKWT(network, config)
# or: router = dmc.DiffusiveWaveIFT(network, config)

# Run simulation
n_timesteps = 100
outlet_Q = np.zeros(n_timesteps)

for t in range(n_timesteps):
    for r in range(10):
        router.set_lateral_inflow(r, runoff[t, r])
    router.route_timestep()
    outlet_Q[t] = router.get_discharge(outlet_reach_id)
```

### Loading mizuRoute Topology

```python
import xarray as xr
import pydmc_route as dmc

# Load topology.nc
ds = xr.open_dataset('topology.nc')
seg_ids = ds['segId'].values
down_seg_ids = ds['downSegId'].values
lengths = ds['length'].values
slopes = ds['slope'].values

# Build network with proper connectivity
network = dmc.Network()
seg_id_to_idx = {int(sid): i for i, sid in enumerate(seg_ids)}

for i in range(len(seg_ids)):
    reach = dmc.Reach()
    reach.id = i
    reach.length = float(lengths[i])
    reach.slope = float(slopes[i])
    reach.manning_n = 0.035
    reach.upstream_junction_id = i
    
    down_id = int(down_seg_ids[i])
    if down_id in seg_id_to_idx:
        reach.downstream_junction_id = seg_id_to_idx[down_id]
    else:
        reach.downstream_junction_id = -1  # Outlet
    
    network.add_reach(reach)

# Create junctions for tributary connections
# ... (see test_routing_with_data.py for full example)

network.build_topology()
```

### Running with Real Data

```bash
# Test with SUMMA output and observations
python python/test_routing_with_data.py --data-dir /path/to/data

# Expected data structure:
# data/
#   settings/mizuRoute/topology.nc
#   simulations/SUMMA/*_timestep.nc
#   observations/**/streamflow.csv
```

## Testing

```bash
# Run C++ tests
cd build
ctest --output-on-failure

# Run Python tests
python python/test_routing_with_data.py --data-dir data --methods mc kwt diffusive
```

## Learnable Parameters

All routers support gradients for Manning's n. The Muskingum-Cunge router additionally supports:

- `width_coef`: Channel width coefficient (W = a × Q^b)
- `width_exp`: Channel width exponent
- `depth_coef`: Channel depth coefficient (D = c × Q^d)  
- `depth_exp`: Channel depth exponent

## Performance

Typical throughput on Apple M1:

| Method | Throughput (timesteps/s) | Notes |
|--------|-------------------------|-------|
| Lag | ~22,000 | Fastest, simple delay |
| MC | ~4,600 | Good balance |
| KWT-Soft | ~4,000 | With gradient support |
| IRF | ~1,100 | Convolution overhead |
| Diffusive | ~800 | Most physics |

## Citation

If you use dRoute in your research, please cite:

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
