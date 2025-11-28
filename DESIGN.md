# Differentiable Muskingum-Cunge Routing (dMC-Route)

## Design Document for C++ Implementation with BMI and Automatic Differentiation

### 1. Overview

This document outlines the architecture for a standalone differentiable Muskingum-Cunge routing library that can:

1. **Route flows** through arbitrary river networks using Muskingum-Cunge physics
2. **Compute gradients** of outputs (streamflow) with respect to parameters (Manning's n, channel geometry) via automatic differentiation
3. **Expose BMI** for seamless coupling with SYMFLUENCE or any NextGen-compatible framework
4. **Enable differentiable optimization** when coupled to upstream differentiable rainfall-runoff models

### 2. Physics: Muskingum-Cunge Routing

#### 2.1 Governing Equation

The Muskingum method relates storage to inflow and outflow:

```
S = K[XI + (1-X)Q]
```

where:
- S = channel storage
- K = storage time constant (travel time)
- X = weighting factor (0 ≤ X ≤ 0.5)
- I = inflow
- Q = outflow

Combined with continuity (dS/dt = I - Q), the discrete solution is:

```
Q(t+Δt) = C₁·I(t+Δt) + C₂·I(t) + C₃·Q(t) + C₄·q_lat·Δx
```

where:
```
C₁ = (Δt - 2KX) / (2K(1-X) + Δt)
C₂ = (Δt + 2KX) / (2K(1-X) + Δt)  
C₃ = (2K(1-X) - Δt) / (2K(1-X) + Δt)
C₄ = 2Δt / (2K(1-X) + Δt)
```

#### 2.2 Muskingum-Cunge Parameters

The Cunge method derives K and X from channel hydraulics:

```
K = Δx / c                          # c = wave celerity
X = 0.5 - Q / (2·c·B·S₀·Δx)        # B = top width, S₀ = bed slope
```

Wave celerity from Manning's equation:
```
c = (5/3)·v = (5/3)·(1/n)·R^(2/3)·S₀^(1/2)
```

#### 2.3 Learnable Parameters

The parameters we want gradients for:
- **Manning's n** (roughness coefficient) - per reach
- **Channel geometry coefficients** (width/depth power laws) - per reach or global

### 3. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              dMC-Route Library                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────────┐ │
│  │   Network    │    │   MC Physics     │    │    Gradient Engine         │ │
│  │   Topology   │    │   (AD-enabled)   │    │    (CoDiPack tape)         │ │
│  │              │    │                  │    │                            │ │
│  │ - Reach DAG  │───▶│ - compute_K_X()  │───▶│ - record forward pass      │ │
│  │ - Junctions  │    │ - route_reach()  │    │ - evaluate adjoints        │ │
│  │ - Ordering   │    │ - route_network()│    │ - extract ∂Q/∂θ            │ │
│  └──────────────┘    └──────────────────┘    └────────────────────────────┘ │
│          │                    │                          │                   │
│          ▼                    ▼                          ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         BMI Interface Layer                             ││
│  │                                                                          ││
│  │  Initialize() | Update() | UpdateUntil() | Finalize()                   ││
│  │  GetValue() | SetValue() | GetValuePtr()                                ││
│  │  GetInputVarNames() | GetOutputVarNames()                               ││
│  │                                                                          ││
│  │  + Extended gradient interface:                                          ││
│  │    GetGradient() | EnableGradients() | GetParameterNames()              ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Key Design Decisions

#### 4.1 AD Library: CoDiPack

**Why CoDiPack over alternatives:**

| Feature | CoDiPack | ADOL-C | autodiff | Enzyme |
|---------|----------|--------|----------|--------|
| Header-only | ✓ | ✗ | ✓ | ✗ |
| Reverse mode | ✓ | ✓ | ✓ | ✓ |
| HPC tested | ✓ (SU2) | ✓ | Limited | ✓ |
| Memory control | ✓ | ✓ | Limited | N/A |
| MPI support | ✓ (MeDiPack) | Limited | ✗ | ✗ |

CoDiPack uses **tape-based reverse mode AD**:
- Forward pass: records operations to tape
- Reverse pass: replays tape to compute gradients
- Memory: O(operations) for tape storage

#### 4.2 Type Abstraction for AD

Use template parameters to switch between:
```cpp
using Real = double;                    // No AD, fastest execution
using Real = codi::RealForward;         // Forward AD (for debugging, small #params)
using Real = codi::RealReverse;         // Reverse AD (for many parameters)
using Real = codi::RealReverseIndex;    // Reduced memory variant
```

#### 4.3 Network Representation

```cpp
struct Reach {
    int id;
    double length;           // Δx [m]
    double slope;            // S₀ [m/m]  
    int upstream_junction;   // node index (-1 if headwater)
    int downstream_junction; // node index
    
    // Learnable parameters (AD types)
    Real manning_n;
    Real width_coef;         // W = a·Q^b (coefficient a)
    Real width_exp;          // W = a·Q^b (exponent b)
    Real depth_coef;         // D = c·Q^d (coefficient c)
    Real depth_exp;          // D = c·Q^d (exponent d)
};

struct Junction {
    int id;
    std::vector<int> upstream_reaches;   // incoming
    std::vector<int> downstream_reaches; // outgoing (usually 1)
};
```

### 5. Core Algorithm

```cpp
template<typename Real>
class MuskingumCungeRouter {
public:
    // Route single reach for one timestep
    Real route_reach(
        int reach_id,
        Real inflow_prev,      // I(t)
        Real inflow_curr,      // I(t+Δt)  
        Real outflow_prev,     // Q(t)
        Real lateral_inflow,   // q_lat·Δx
        double dt
    ) {
        Reach& r = reaches_[reach_id];
        
        // Estimate discharge for hydraulics (use previous outflow)
        Real Q_ref = outflow_prev;
        
        // Compute hydraulic geometry
        Real width = r.width_coef * pow(Q_ref, r.width_exp);
        Real depth = r.depth_coef * pow(Q_ref, r.depth_exp);
        Real area = width * depth;
        Real R_h = area / (width + 2*depth);  // hydraulic radius
        
        // Wave celerity from Manning
        Real velocity = (1.0/r.manning_n) * pow(R_h, 2.0/3.0) * sqrt(r.slope);
        Real celerity = (5.0/3.0) * velocity;
        
        // Muskingum K and X
        Real K = r.length / celerity;
        Real X = 0.5 - Q_ref / (2.0 * celerity * width * r.slope * r.length);
        X = max(Real(0.0), min(Real(0.5), X));  // Clamp to valid range
        
        // Routing coefficients
        Real denom = 2.0*K*(1.0-X) + dt;
        Real C1 = (dt - 2.0*K*X) / denom;
        Real C2 = (dt + 2.0*K*X) / denom;
        Real C3 = (2.0*K*(1.0-X) - dt) / denom;
        Real C4 = 2.0*dt / denom;
        
        // Route
        Real outflow = C1*inflow_curr + C2*inflow_prev + C3*outflow_prev + C4*lateral_inflow;
        return max(Real(0.0), outflow);  // Non-negative flow
    }
    
    // Route entire network for one timestep (topologically sorted)
    void route_network(double dt) {
        for (int reach_id : topological_order_) {
            // Gather upstream contributions at junction
            Real inflow = compute_junction_inflow(reach_id);
            
            // Route through reach
            outflow_curr_[reach_id] = route_reach(
                reach_id,
                inflow_prev_[reach_id],
                inflow,
                outflow_prev_[reach_id],
                lateral_[reach_id],
                dt
            );
        }
        // Swap state
        std::swap(inflow_prev_, inflow_curr_);
        std::swap(outflow_prev_, outflow_curr_);
    }
};
```

### 6. Gradient Computation

#### 6.1 Recording Phase (Forward Pass)

```cpp
void forward_with_tape(int num_timesteps, double dt) {
    using Tape = codi::RealReverse::Tape;
    Tape& tape = codi::RealReverse::getTape();
    
    tape.setActive();
    
    // Register parameters as inputs
    for (auto& reach : reaches_) {
        tape.registerInput(reach.manning_n);
        tape.registerInput(reach.width_coef);
        // ... etc
    }
    
    // Run forward simulation
    for (int t = 0; t < num_timesteps; ++t) {
        route_network(dt);
    }
    
    // Register outputs
    for (int i : gauge_reach_ids_) {
        tape.registerOutput(outflow_curr_[i]);
    }
    
    tape.setPassive();
}
```

#### 6.2 Adjoint Evaluation (Backward Pass)

```cpp
void compute_gradients(const std::vector<double>& dL_dQ) {
    using Tape = codi::RealReverse::Tape;
    Tape& tape = codi::RealReverse::getTape();
    
    // Seed adjoints with loss gradients
    for (size_t i = 0; i < gauge_reach_ids_.size(); ++i) {
        outflow_curr_[gauge_reach_ids_[i]].setGradient(dL_dQ[i]);
    }
    
    // Reverse pass
    tape.evaluate();
    
    // Extract parameter gradients
    for (auto& reach : reaches_) {
        grad_manning_n_[reach.id] = reach.manning_n.getGradient();
        grad_width_coef_[reach.id] = reach.width_coef.getGradient();
        // ... etc
    }
    
    tape.reset();
}
```

### 7. BMI Interface

```cpp
class BmiMuskingumCunge : public bmi::Bmi {
public:
    // Standard BMI
    void Initialize(std::string config_file) override;
    void Update() override;
    void UpdateUntil(double time) override;
    void Finalize() override;
    
    std::string GetComponentName() override { return "dMC-Route"; }
    
    std::vector<std::string> GetInputVarNames() override {
        return {"lateral_inflow", "upstream_boundary"};
    }
    
    std::vector<std::string> GetOutputVarNames() override {
        return {"discharge", "velocity", "depth"};
    }
    
    void GetValue(std::string name, void* dest) override;
    void SetValue(std::string name, void* src) override;
    
    // Extended interface for differentiable coupling
    std::vector<std::string> GetParameterNames();
    void GetParameterGradient(std::string name, void* dest);
    void SetLossGradient(std::string name, void* src);
    void EnableGradientRecording(bool enable);
    void ComputeGradients();
};
```

### 8. Configuration File Format (YAML)

```yaml
dmc_route:
  network:
    topology_file: "network.geojson"  # or CSV
    
  time:
    dt: 3600.0          # timestep [s]
    
  parameters:
    # Global defaults (can be overridden per-reach)
    manning_n: 0.035
    width_coef: 7.2
    width_exp: 0.5
    depth_coef: 0.27
    depth_exp: 0.3
    
  differentiation:
    enabled: true
    mode: "reverse"     # "forward" | "reverse" | "none"
    
  output:
    gauge_reaches: [101, 203, 456]  # reach IDs to output
```

### 9. Build System (CMake)

```cmake
cmake_minimum_required(VERSION 3.14)
project(dmc_route VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(DMC_ENABLE_AD "Enable automatic differentiation" ON)
option(DMC_BUILD_PYTHON "Build Python bindings" OFF)

# Dependencies
find_package(yaml-cpp REQUIRED)
include(FetchContent)

# CoDiPack (header-only)
FetchContent_Declare(
    codipack
    GIT_REPOSITORY https://github.com/SciCompKL/CoDiPack.git
    GIT_TAG v2.2.0
)
FetchContent_MakeAvailable(codipack)

# BMI
FetchContent_Declare(
    bmi_cxx
    GIT_REPOSITORY https://github.com/csdms/bmi-cxx.git
    GIT_TAG master
)
FetchContent_MakeAvailable(bmi_cxx)

# Library
add_library(dmc_route
    src/network.cpp
    src/router.cpp
    src/bmi_adapter.cpp
)

target_include_directories(dmc_route PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${codipack_SOURCE_DIR}/include
    ${bmi_cxx_SOURCE_DIR}
)

target_link_libraries(dmc_route PUBLIC yaml-cpp)

if(DMC_ENABLE_AD)
    target_compile_definitions(dmc_route PUBLIC DMC_USE_CODIPACK)
endif()

# Executable for standalone testing
add_executable(dmc_route_run src/main.cpp)
target_link_libraries(dmc_route_run dmc_route)
```

### 10. Integration with SYMFLUENCE

#### 10.1 As a BMI Component

```python
# Python orchestration (SYMFLUENCE)
import ctypes
from bmi import Bmi

# Load compiled library
lib = ctypes.CDLL("libdmc_route.so")
router = create_bmi_model(lib)

# Initialize
router.initialize("config.yaml")

# Coupling loop
for t in range(n_timesteps):
    # Get runoff from upstream model (e.g., SUMMA via BMI)
    lateral = summa.get_value("runoff")
    router.set_value("lateral_inflow", lateral)
    
    # Route
    router.update()
    
    # Get routed discharge
    Q = router.get_value("discharge")
```

#### 10.2 End-to-End Differentiable Training

```python
# Pseudocode for differentiable optimization
# Upstream model (SUMMA/HBV) + Routing (dMC-Route)

def forward_pass(params, forcing, observations):
    # Set parameters
    runoff_model.set_parameters(params['runoff'])
    router.set_parameters(params['routing'])
    
    # Enable gradient recording
    router.enable_gradient_recording(True)
    
    # Run simulation
    Q_sim = []
    for t, forcing_t in enumerate(forcing):
        runoff = runoff_model.forward(forcing_t)
        router.set_value("lateral_inflow", runoff)
        router.update()
        Q_sim.append(router.get_value("discharge"))
    
    # Compute loss
    loss = mse(Q_sim, observations)
    return loss, Q_sim

def backward_pass(loss, Q_sim, observations):
    # Compute dL/dQ
    dL_dQ = 2 * (Q_sim - observations) / len(Q_sim)
    
    # Backpropagate through router
    router.set_loss_gradient("discharge", dL_dQ)
    router.compute_gradients()
    
    # Get parameter gradients
    grad_n = router.get_parameter_gradient("manning_n")
    grad_w = router.get_parameter_gradient("width_coef")
    
    return {'manning_n': grad_n, 'width_coef': grad_w, ...}
```

### 11. Testing Strategy

1. **Unit tests**: Individual reach routing against analytical solutions
2. **Network tests**: Compare with mizuRoute/MC-Flood for same network
3. **Gradient verification**: Finite differences vs AD gradients
4. **Conservation tests**: Mass balance over network

### 12. Performance Considerations

| Concern | Mitigation |
|---------|------------|
| Tape memory | Use checkpointing for long simulations |
| Many reaches | Parallelism via OpenMP (CoDiPack compatible) |
| Real-time | Compile without AD for operational mode |
| Large networks | Graph partitioning for distributed memory |

### 13. Timeline Estimate

| Phase | Effort | Description |
|-------|--------|-------------|
| Core routing | 2-3 days | MC physics, network traversal |
| CoDiPack integration | 3-4 days | Templating, tape management |
| BMI wrapper | 2-3 days | Standard + extended interface |
| Config & I/O | 2 days | YAML, GeoJSON/CSV network parsing |
| Testing | 3-4 days | Unit, integration, gradient verification |
| **Total** | **~2-3 weeks** | For functional prototype |

### 14. Advanced Features (v0.4.0)

#### 14.1 Generalized Hydraulic Geometry

Replaces hardcoded "5/3 wide-rectangular" assumption with:

**Power-Law (Leopold-Maddock):**
```
c = (5/3 - 2b/3) × v
```
where b is the width exponent. Reduces to 5/3×v when b=0 (rectangular).

**Trapezoidal Channels:**
- Newton-Raphson solver for depth from Manning's equation
- Analytical celerity correction factor

**Compound Channels:**
- Main channel + floodplain with divided conveyance
- Automatic switching at bankfull depth

See `channel_geometry.hpp` for implementation.

#### 14.2 Fixed-Count Sub-stepping (AD-Safe)

```cpp
config.fixed_substepping = true;
config.num_substeps = 4;
```

Divides each timestep into fixed sub-steps to ensure numerical stability on short reaches without introducing control-flow discontinuities that break AD. Inflows are linearly interpolated across sub-steps.

#### 14.3 Soft-Masked IRF Convolution

Solves the **integer loop-bound problem** for IRF routing:

Traditional approach:
```cpp
for (int i = 0; i < kernel_length; ++i)  // Length varies with parameters!
```

Soft-masked approach:
```cpp
for (int i = 0; i < MAX_LENGTH; ++i)  // Fixed loop
    weight[i] = base_weight[i] * sigmoid((T_cutoff - t_i) * steepness)
```

The sigmoid mask smoothly transitions from 1→0, allowing gradients to flow through changes in travel time.

#### 14.4 IFT Adjoints for Diffusive Wave

Uses Implicit Function Theorem instead of recording solver operations:

**Forward:** Solve `A × Q_new = B × Q_old + b` using Thomas algorithm

**Adjoint:** Solve `A^T × λ = dL/dQ_new` (same algorithm, swapped diagonals)

**Gradient:** `dL/dn = -λ^T × (∂(Ax-b)/∂n)`

Benefits:
- Exact gradients through implicit solver
- Memory: O(n) instead of O(n × iterations)
- Works for Crank-Nicolson and other implicit schemes

See `DiffusiveWaveIFT` class in `advanced_routing.hpp`.

#### 14.5 Soft-Gated KWT (Differentiable)

Replaces binary parcel crossing logic with smooth probabilistic flux:

```cpp
P(exit) = sigmoid((position - L) × steepness / spread)
flux = volume × P(exit)
```

Each parcel has:
- Position (center of wave)
- Spread (Gaussian σ, increases with dispersion)
- Remaining fraction (tracks cumulative exit)

Gradient propagates through:
```
dQ_out/dn = Σ_parcels [dP/dpos × dpos/dc × dc/dn] × volume/dt
```

#### 14.6 Parallel Graph Traversal

Uses **graph coloring** to identify independent reaches:

```cpp
GraphColoring coloring(network);
for (const auto& group : coloring.get_color_groups()) {
    #pragma omp parallel for
    for (int reach_id : group) {
        // Process in parallel - no dependencies within group
    }
}
```

Typically achieves 3-5 color groups for dendritic networks, enabling ~3-5× parallel speedup.

#### 14.7 Revolve Checkpointing

Implements **binomial checkpointing** (Griewank & Walther algorithm) for memory-efficient adjoints:

```cpp
CheckpointedRouter<MuskingumCungeRouter> router(network, num_checkpoints, config);
router.route(26000);  // Multi-year simulation
router.compute_gradients(gauges, dL_dQ);  // Uses O(checkpoints) memory
```

Memory-time tradeoff:
- With C checkpoints for T steps: O(T × log(T)) recomputation
- Without checkpointing: O(T) memory (tape storage)

#### 14.8 Full State Serialization

```cpp
RouterState state;
state.save("checkpoint.bin");  // Binary serialization
auto loaded = RouterState::load("checkpoint.bin");
```

Stores:
- Current time
- Inflow/outflow states for all reaches
- Internal buffers (IRF history, KWT parcels)

Use cases:
- Warm starts for operational forecasting
- Splitting long training windows
- Checkpoint/restart for HPC jobs

### 15. Routing Method Comparison

| Method | Full AD | Gradients | Accuracy | Speed | Use Case |
|--------|---------|-----------|----------|-------|----------|
| Muskingum-Cunge | ✅ | All 3 params | Good | Fast | Production calibration |
| IRF | ✅* | manning_n | Good | Fast | Fast calibration |
| IRF (soft-masked) | ✅ | manning_n | Good | Medium | When kernel length varies |
| Diffusive Wave | ⚠️ | manning_n | Best | Slow | High-accuracy physics |
| Diffusive-IFT | ✅ | manning_n | Best | Slow | Exact adjoint needed |
| Lag | ⚠️ | manning_n | Basic | Fast | Baseline comparison |
| KWT | ❌ | None | Good | Fast | mizuRoute compatibility |
| KWT-Soft | ✅ | manning_n | Good | Medium | Differentiable Lagrangian |

Legend:
- ✅ = Full AD support
- ⚠️ = Analytical approximation
- ❌ = Not differentiable
- *IRF uses soft-masking when `irf_soft_mask=true`

### 16. References

1. Cunge, J.A. (1969). On the subject of a flood propagation computation method. *J. Hydraul. Res.*, 7(2), 205-230.
2. Bindas et al. (2024). Improving River Routing Using a Differentiable Muskingum-Cunge Model. *WRR*.
3. Sagebaum et al. (2019). High-Performance Derivative Computations using CoDiPack. *ACM TOMS*.
4. Hutton et al. (2020). The Basic Model Interface 2.0. *JOSS*.
5. Griewank, A., & Walther, A. (2000). Algorithm 799: Revolve. *ACM TOMS*, 26(1), 19-45.
6. Mizukami et al. (2016). mizuRoute version 1: a river network routing tool. *GMD*, 9(6), 2223-2238.
