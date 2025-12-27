# dRoute: Differentiable River Routing Library

A differentiable river routing library for hydrological modeling. dRoute implements multiple routing methods with automatic differentiation support, enabling gradient-based parameter optimization for seamless integration with machine learning workflows.

## Features

- **6 Routing Methods**: From simple lag routing to full Saint-Venant equations
- **Dual AD Backends**: CoDiPack (tape-based) and Enzyme (source-to-source) 
- **Fast Optimization**: Gradient descent calibration of Manning's n in seconds
- **Network Topology**: Full support for river networks with tributaries and confluences
- **mizuRoute Compatible**: Reads standard `topology.nc` files
- **PyTorch Integration**: Use dRoute gradients in ML training loops

## Routing Methods

| Method | Class | Physics | Speed | Use Case |
|--------|-------|---------|-------|----------|
| **Muskingum-Cunge** | `MuskingumCungeRouter` | Kinematic + diffusion approx | ~4,500/s | Production routing |
| **Lag** | `LagRouter` | Time delay buffer | ~20,000/s | Baseline comparison |
| **IRF** | `IRFRouter` | Gamma unit hydrograph | ~1,000/s | Fast calibration |
| **KWT-Soft** | `SoftGatedKWT` | Kinematic wave tracking | ~4,000/s | Differentiable Lagrangian |
| **Diffusive Wave** | `DiffusiveWaveIFT` | Diffusion wave PDE | ~3,000/s | Flood wave attenuation |
| **Saint-Venant** | `SaintVenantRouter` | Full shallow water eqs | ~100/s | High-fidelity benchmark |

## Quick Start

### Installation

```bash
git clone https://github.com/DarriEy/dRoute.git
cd dRoute
mkdir build && cd build
cmake .. -DDMC_BUILD_PYTHON=ON
make -j4
```

### Test Installation

```bash
cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python
python -c "import droute; print(f'dRoute v{droute.__version__} loaded')"
```

### Run with Sample Data

```bash
# Forward pass comparison (all methods)
PYTHONPATH=build/python python python/test_routing_with_data.py --data-dir data

# Fast optimization with Enzyme kernels (30 epochs in ~30s)
PYTHONPATH=build/python python python/test_routing_with_data.py --data-dir data --optimize --fast

# Include Saint-Venant high-fidelity benchmark
PYTHONPATH=build/python python python/test_routing_with_data.py --data-dir data --sve
```

### Download Sample Data

The example dataset is hosted as a GitHub release asset (v0.5.0).

```bash
python scripts/download_data.py
```

## Usage

### Basic Python API

```python
import droute
import numpy as np

# Load network from mizuRoute topology file
network = droute.Network()
droute.load_topology(network, "data/settings/dRoute/topology.nc")

# Or build programmatically
network = droute.Network()
for i in range(10):
    reach = droute.Reach()
    reach.id = i
    reach.length = 5000.0  # meters
    reach.slope = 0.001
    reach.manning_n = 0.035
    reach.downstream_id = i + 1 if i < 9 else -1  # -1 = outlet
    network.add_reach(reach)
network.build_topology()

# Create router
config = droute.RouterConfig()
config.dt = 3600.0  # 1-hour timestep

router = droute.MuskingumCungeRouter(network, config)

# Run simulation
runoff = np.random.rand(100, 10) * 0.001  # (timesteps, reaches)
outlet_Q = []

for t in range(100):
    for r in range(10):
        router.set_lateral_inflow(r, runoff[t, r])
    router.route_timestep()
    outlet_Q.append(router.get_discharge(9))
```

### Gradient Computation with CoDiPack

```python
# Enable gradient recording
config.enable_gradients = True
router = droute.MuskingumCungeRouter(network, config)

router.start_recording()

# Forward pass - record outputs at each timestep
sim = np.zeros(n_timesteps)
for t in range(n_timesteps):
    for r in range(n_reaches):
        router.set_lateral_inflow(r, runoff[t, r])
    router.route_timestep()
    router.record_output(outlet_id)  # Record for AD tape
    sim[t] = router.get_discharge(outlet_id)

router.stop_recording()

# Compute gradients for MSE loss: dL/dQ = 2(sim - obs) / n
dL_dQ = (2.0 / n_timesteps) * (sim - observed)
router.compute_gradients_timeseries(outlet_id, dL_dQ.tolist())

# Get parameter gradients
grads = router.get_gradients()
print(f"dLoss/d(manning_n) for reach 0: {grads['reach_0_manning_n']}")
```

### Fast Optimization with Enzyme

```python
# Create Enzyme router (no tape overhead, ~5x faster)
router = droute.enzyme.EnzymeRouter(network, dt=3600.0, num_substeps=4, method=0)

# Methods: 0=MC, 1=Lag, 2=IRF, 3=KWT, 4=Diffusive

# Optimize Manning's n using built-in gradient descent
result = droute.enzyme.optimize(
    router,
    runoff,           # (n_timesteps, n_reaches)
    observed,         # (n_timesteps,)
    outlet_reach=0,
    n_epochs=30,
    lr=0.1,
    verbose=True
)

print(f"Final NSE: {result['nse']:.3f}")
print(f"Optimized Manning's n: {result['optimized_manning_n']}")
```

### Saint-Venant High-Fidelity Benchmark

```python
# Configure SVE solver
config = droute.SaintVenantConfig()
config.dt = 3600.0
config.n_nodes = 10          # Spatial nodes per reach
config.initial_depth = 0.5   # Initial water depth [m]
config.rel_tol = 1e-4        # CVODES tolerances
config.abs_tol = 1e-6

# Create router (uses SUNDIALS CVODES if available)
router = droute.SaintVenantRouter(network, config)

# Run simulation - SVE also provides water depth!
for t in range(n_timesteps):
    for r in range(n_reaches):
        router.set_lateral_inflow(r, runoff[t, r])
    router.route_timestep()
    Q[t] = router.get_discharge(outlet_id)
    h[t] = router.get_depth(outlet_id)
```

### PyTorch Integration

```python
import torch

# Initialize parameters
log_manning_n = torch.zeros(n_reaches, requires_grad=True, dtype=torch.float64)
optimizer = torch.optim.Adam([log_manning_n], lr=0.1)

for epoch in range(30):
    optimizer.zero_grad()
    
    # Update network parameters
    manning_n = torch.exp(log_manning_n)
    for r in range(n_reaches):
        network.get_reach(r).manning_n = manning_n[r].item()
    
    # Forward pass with CoDiPack
    router.start_recording()
    sim = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        for r in range(n_reaches):
            router.set_lateral_inflow(r, runoff[t, r])
        router.route_timestep()
        router.record_output(outlet_id)
        sim[t] = router.get_discharge(outlet_id)
    router.stop_recording()
    
    # Compute loss and gradients
    loss = np.mean((sim - observed) ** 2)
    dL_dQ = (2.0 / n_timesteps) * (sim - observed)
    router.compute_gradients_timeseries(outlet_id, dL_dQ.tolist())
    
    # Transfer gradients to PyTorch (with chain rule for log transform)
    grads = router.get_gradients()
    grad_n = np.array([grads.get(f"reach_{r}_manning_n", 0.0) for r in range(n_reaches)])
    log_manning_n.grad = torch.tensor(grad_n * manning_n.detach().numpy())
    
    optimizer.step()
    print(f"Epoch {epoch}: MSE={loss:.4f}")
```

## Build Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `DMC_BUILD_PYTHON` | OFF | Build Python bindings |
| `DMC_USE_CODIPACK` | ON | Enable CoDiPack AD |
| `DMC_ENABLE_ENZYME` | OFF | Enable Enzyme AD backend |
| `DMC_ENABLE_SUNDIALS` | OFF | Enable SUNDIALS for SVE solver |
| `DMC_ENABLE_NETCDF` | OFF | Enable NetCDF topology I/O |
| `DMC_ENABLE_OPENMP` | OFF | Enable OpenMP parallelization |
| `DMC_BUILD_TESTS` | ON | Build C++ test suite |

### Build with All Features

```bash
cmake .. \
    -DDMC_BUILD_PYTHON=ON \
    -DDMC_ENABLE_ENZYME=ON \
    -DDMC_ENABLE_SUNDIALS=ON \
    -DSUNDIALS_ROOT=/path/to/sundials/install \
    -DDMC_ENABLE_NETCDF=ON

make -j4
```

### macOS (Apple Silicon)

```bash
brew install cmake netcdf sundials

cmake .. -DDMC_BUILD_PYTHON=ON -DDMC_ENABLE_SUNDIALS=ON
make -j$(sysctl -n hw.ncpu)
```

### Linux HPC

```bash
module load cmake gcc python sundials netcdf

cmake .. \
    -DDMC_BUILD_PYTHON=ON \
    -DDMC_ENABLE_SUNDIALS=ON \
    -DCMAKE_CXX_COMPILER=g++

make -j8
```

## Data Format

### topology.nc (mizuRoute compatible)

| Variable | Dimensions | Description |
|----------|------------|-------------|
| `segId` | (seg) | Segment IDs |
| `downSegId` | (seg) | Downstream segment ID (-1 for outlet) |
| `length` | (seg) | Reach length [m] |
| `slope` | (seg) | Channel slope [m/m] |
| `hruId` | (hru) | HRU IDs |
| `hruToSegId` | (hru) | HRU to segment mapping |
| `area` | (hru) | HRU area [m²] |

### Directory Structure

```
data/
├── settings/
│   └── dRoute/
│       └── topology.nc          # River network
├── simulations/
│   └── SUMMA/
│       └── run_1_timestep.nc    # Runoff from SUMMA
└── observations/
    └── streamflow/
        └── streamflow.csv       # Observed discharge
```

## Learnable Parameters

All routers support gradients for:
- `manning_n`: Manning's roughness coefficient

Muskingum-Cunge additionally supports:
- `width_coef`, `width_exp`: W = a × Q^b
- `depth_coef`, `depth_exp`: D = c × Q^d

## Architecture

```
dRoute/
├── include/dmc/
│   ├── router.hpp              # MuskingumCungeRouter with CoDiPack AD
│   ├── advanced_routing.hpp    # IRF, KWT, Diffusive routers  
│   ├── kernels_enzyme.hpp      # Enzyme-compatible kernels (all 5 methods)
│   ├── unified_router.hpp      # EnzymeRouter wrapper
│   ├── saint_venant_router.hpp # Full SVE with SUNDIALS CVODES
│   ├── network.hpp             # Network topology
│   └── types.hpp               # AD type definitions
├── python/
│   ├── bindings.cpp            # pybind11 bindings
│   └── test_routing_with_data.py
├── tests/                      # C++ test suite
└── CMakeLists.txt
```

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15+
- pybind11 (auto-downloaded if not found)
- CoDiPack (auto-downloaded)
- Optional: Enzyme, SUNDIALS, NetCDF-C, OpenMP

## Testing

```bash
# C++ tests
cd build && ctest --output-on-failure

# Python tests
cd ..
PYTHONPATH=build/python python python/test_routing_with_data.py --data-dir data

# Full test suite with optimization and SVE
PYTHONPATH=build/python python python/test_routing_with_data.py \
    --data-dir data --optimize --fast --sve
```

## Citation

```bibtex
@software{dRoute2025,
  title={dRoute: Differentiable River Routing Library},
  author={Eythorsson, Darri},
  year={2024},
  url={https://github.com/DarriEy/dRoute}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [CoDiPack](https://github.com/SciCompKL/CoDiPack) - Tape-based automatic differentiation
- [Enzyme](https://enzyme.mit.edu/) - Source-to-source AD compiler plugin
- [SUNDIALS](https://computing.llnl.gov/projects/sundials) - Implicit ODE solvers
- [SUMMA](https://github.com/NCAR/summa) & [mizuRoute](https://github.com/NCAR/mizuRoute) - Hydrological modeling inspiration
