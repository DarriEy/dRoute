# dRoute: Differentiable River Routing Library

A differentiable routing library for hydrological modeling and machine learning applications. Implements multiple routing methods with automatic differentiation support via CoDiPack.

## Routing Methods

| Method | Gradients | Speed | Use Case |
|--------|-----------|-------|----------|
| Muskingum-Cunge | Full AD (5 params) | Fast | Production calibration |
| IRF | Full AD (soft-masked) | Fast | Fast calibration |
| Diffusive Wave | Analytical | Medium | High-accuracy physics |
| Diffusive-IFT | Exact (IFT) | Medium | Exact gradients needed |
| Lag | Analytical (weak) | Fast | Baseline only |
| KWT | None | Fast | mizuRoute compatibility |
| KWT-Soft | Full AD | Medium | Differentiable Lagrangian |

## Installation

### From Source (C++)

```bash
git clone https://github.com/your-org/dmc-route.git
cd dmc-route
mkdir build && cd build
cmake .. -DDMC_ENABLE_NETCDF=ON
make -j4
```

### Python Bindings

```bash
cd build
cmake .. -DDMC_BUILD_PYTHON=ON
make pydmc_route

# Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/python
```

Or install via pip:

```bash
pip install .
```

## Quick Start

### C++ API

```cpp
#include <dmc/router.hpp>
#include <dmc/network.hpp>

using namespace dmc;

// Create network
Network net;
// ... add reaches and junctions ...
net.build_topology();

// Configure router
RouterConfig config;
config.dt = 3600.0;
config.enable_gradients = true;

// Create router
MuskingumCungeRouter router(net, config);

// Run simulation with gradient recording
router.start_recording();
for (int t = 0; t < num_timesteps; ++t) {
    set_lateral_inflows(net, runoff[t]);
    router.route_timestep();
}
router.stop_recording();

// Compute gradients
router.compute_gradients({outlet_id}, {1.0});

// Get gradients
auto grads = router.get_gradients();
```

### Python API

```python
import pydmc_route as dmc

# Create network
network = dmc.Network()
# ... add reaches and junctions ...
network.build_topology()

# Configure router
config = dmc.RouterConfig()
config.dt = 3600.0
config.enable_gradients = True

# Create router
router = dmc.MuskingumCungeRouter(network, config)

# Run simulation
router.start_recording()
for t in range(num_timesteps):
    network.set_lateral_inflows(runoff[t, :])
    router.route_timestep()
router.stop_recording()

# Compute gradients
router.compute_gradients([outlet_id], [1.0])
grads = network.get_grad_manning_n_all()
```


## Learnable Parameters

The Muskingum-Cunge router supports gradients for:
- `manning_n`: Manning's roughness coefficient
- `width_coef`: Width coefficient (W = a × Q^b)
- `depth_coef`: Depth coefficient (D = c × Q^d)

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `DMC_ENABLE_NETCDF` | OFF | Enable NetCDF for topology I/O |
| `DMC_ENABLE_OPENMP` | OFF | Enable OpenMP parallelization |
| `DMC_BUILD_PYTHON` | OFF | Build Python bindings |
| `DMC_BUILD_TESTS` | ON | Build test suite |
| `DMC_BUILD_SHARED` | OFF | Build shared library |

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15+
- pybind11 (auto-downloaded if not found)
- CoDiPack (auto-downloaded)
- Optional: NetCDF-C, OpenMP
