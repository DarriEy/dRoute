# dMC-Route: Routing Methods Reference

This document describes the routing methods available in dMC-Route, their governing equations, numerical schemes, and gradient capabilities.

## Overview

| Method | Command | Gradients | Accuracy | Speed | Use Case |
|--------|---------|-----------|----------|-------|----------|
| Muskingum-Cunge | `-r muskingum` | Full AD (5 params) | Good | Fast | Production calibration |
| IRF | `-r irf` | Full AD (soft-masked) | Good | Fast | Fast calibration |
| Diffusive Wave | `-r diffusive` | Analytical approx | Best | Slow | High-accuracy physics |
| Diffusive-IFT | `-r diffusive-ift` | Exact IFT | Best | Slow | Exact gradients needed |
| Lag | `-r lag` | Analytical (weak) | Basic | Fast | Baseline, NOT for calibration |
| KWT | `-r kwt` | None | Good | Fast | mizuRoute compatibility |
| KWT-Soft | `-r kwt-soft` | Full AD (soft-gated) | Good | Medium | Differentiable Lagrangian |

---

## 1. Muskingum-Cunge (Default)

### Governing Equation
Storage-outflow relationship with Cunge's hydraulic derivation:
```
S = K[XI + (1-X)Q]
dS/dt = I - Q
```

### Discrete Solution
```
Q(t+Δt) = C₁·I(t+Δt) + C₂·I(t) + C₃·Q(t) + C₄·q_lat·Δx
```

Coefficients:
```
K = Δx / c           (c = wave celerity = 5/3 × v)
X = 0.5 - Q / (2·c·B·S₀·Δx)

C₁ = (Δt - 2KX) / (2K(1-X) + Δt)
C₂ = (Δt + 2KX) / (2K(1-X) + Δt)
C₃ = (2K(1-X) - Δt) / (2K(1-X) + Δt)
C₄ = 2Δt / (2K(1-X) + Δt)
```

### Numerical Scheme
- Explicit Muskingum-Cunge with fixed sub-stepping
- Sub-steps: 4 by default (AD-safe, no control flow branching)
- Smooth clamping of X to [0, 0.5] when gradients enabled

### Differentiability
- **Full AD** via CoDiPack
- Parameters: `manning_n`, `width_coef`, `width_exp`, `depth_coef`, `depth_exp`
- Uses `exp(y × log(x))` form for power-law geometry (AD-safe exponents)

### Recommended Use
- Production calibration
- General-purpose routing
- When all 5 geometry parameters need gradients

---

## 2. IRF (Impulse Response Function)

### Governing Equation
Convolution with gamma unit hydrograph:
```
Q_out(t) = ∑_τ k(τ; n, geom) × I(t - τ)
```

Gamma kernel:
```
k(t) = (1/Γ(α)θ^α) × t^(α-1) × exp(-t/θ)
```
where α = shape parameter, θ = scale = travel_time / α

### Numerical Scheme
- Direct convolution with stored inflow history
- Soft-masked kernel (sigmoid) for differentiable kernel length
- Fixed maximum kernel size (200 timesteps default)

### Differentiability
- **Full AD** when `irf_soft_mask = true`
- Travel time computed from manning_n at each timestep
- Sigmoid mask: `mask[i] = σ((T_cutoff - t_i) × steepness)`
- Kernel weights recomputed on-the-fly for gradient flow

### Recommended Use
- Fast calibration of travel time
- When detailed wave physics not critical
- Unit hydrograph-based applications

---

## 3. Diffusive Wave

### Governing Equation
Diffusive wave approximation to Saint-Venant equations:
```
∂Q/∂t + c·∂Q/∂x = D·∂²Q/∂x² + q_lat
```
where:
- c = wave celerity [m/s]
- D = diffusion coefficient = Q / (2·B·S₀) [m²/s]
- q_lat = lateral inflow per unit length [m²/s]

### Numerical Scheme
- Explicit finite difference
- Upwind advection, central diffusion
- Adaptive sub-stepping for CFL stability
- Distributed lateral inflow as source term

### Differentiability
- **Analytical approximation** for manning_n
- Approximation: `dQ_out/dn ≈ -advection_effect / n`
- Limited accuracy due to approximation

### Recommended Use
- High-accuracy flood wave physics
- When wave attenuation is important
- Research applications

---

## 4. Diffusive Wave IFT

### Governing Equation
Same as Diffusive Wave, but with implicit solver.

### Numerical Scheme
- Crank-Nicolson implicit scheme (θ = 0.5)
- Tridiagonal system solved via Thomas algorithm
- IFT (Implicit Function Theorem) for adjoint computation

### IFT Adjoint Derivation
For system `A × Q_new = B × Q_old + b`:
1. Forward: Solve tridiagonal system
2. Store: Matrix coefficients and sensitivities
3. Adjoint: Solve `A^T × λ = dL/dQ_new`
4. Gradient: `dL/dn = -λ^T × (∂(Ax-b)/∂n)`

### Differentiability
- **Exact gradients** via IFT
- No AD tape overhead for solver iterations
- Memory: O(n) instead of O(n × iterations)

### Recommended Use
- When exact gradients through implicit solver needed
- Long-duration simulations where tape memory matters
- Research into adjoint methods

---

## 5. Lag Router

### Governing Equation
Pure delay:
```
Q_out(t) = Q_in(t - τ)
```
where τ = travel_time = length / velocity

### Numerical Scheme
- FIFO buffer with integer lag (# timesteps)
- lag_steps = round(travel_time / dt)

### Differentiability
- **NOT recommended for calibration**
- Integer lag means step-function response to parameter changes
- Analytical approximation provided but weak

### Recommended Use
- Simple delay for comparison/baseline
- When physics-based routing not needed
- Forward simulation only

---

## 6. KWT (Kinematic Wave Tracking)

### Governing Equation
Lagrangian tracking of wave parcels:
```
dx/dt = c(Q)      (parcel advection)
c = (5/3) × v     (kinematic wave celerity)
```

### Numerical Scheme
- Wave parcels track volume through reach
- Continuous wave segments (not point masses)
- Fraction-based exit computation

### Differentiability
- **Non-differentiable**
- Binary parcel crossing decisions
- Use for mizuRoute compatibility only

### Recommended Use
- mizuRoute compatibility
- Forward diagnostic runs
- When gradients not needed

---

## 7. KWT-Soft (Soft-Gated KWT)

### Governing Equation
Same physics as KWT, but with probabilistic flux:
```
P(exit) = σ((position - length) × steepness / spread)
flux = volume × P(exit)
```

### Numerical Scheme
- Wave parcels with spatial spread (Gaussian σ)
- Sigmoid-gated exit probability
- Gradients through chain rule

### Gradient Flow
```
dQ_out/dn = Σ_parcels [dP/dpos × dpos/dc × dc/dn] × volume/dt
```

### Steepness Annealing
- Start with low steepness (1.0) for smooth gradients
- Increase to high steepness (20.0) for sharp physics
- Use `router.set_steepness()` during training

### Differentiability
- **Full AD** through soft gate
- Parameter: manning_n (via celerity)

### Recommended Use
- Differentiable Lagrangian routing
- When parcel-based physics preferred over Eulerian
- Research into hybrid methods

---

## Configuration Options

```cpp
struct RouterConfig {
    // Basic
    double dt = 3600.0;           // Timestep [s]
    bool enable_gradients = true; // Enable AD
    
    // AD Safety
    bool use_smooth_bounds = true;   // Smooth clamp/max/min
    double smooth_epsilon = 1e-6;    // Smoothing parameter
    
    // Sub-stepping (MC)
    bool fixed_substepping = true;   // AD-safe
    int num_substeps = 4;
    
    // IRF
    int irf_max_kernel_size = 200;
    double irf_shape_param = 2.5;
    bool irf_soft_mask = true;
    double irf_mask_steepness = 10.0;
    
    // Diffusive Wave
    int dw_num_nodes = 10;
    bool dw_use_ift_adjoint = true;
    bool dw_distributed_lateral = true;
    
    // KWT-Soft
    double kwt_gate_steepness = 5.0;
    bool kwt_anneal_steepness = false;
    double kwt_steepness_min = 1.0;
    double kwt_steepness_max = 20.0;
};
```

---

## Gradient Verification

Run the test suite to verify gradients:
```bash
cd build && ./test_gradient_verification
```

The suite tests:
1. FD vs AD for all parameters
2. Gradient sign consistency
3. Downstream gradient amplification
4. Mass balance
5. Lateral inflow handling
6. Multiple gauge scenarios

---

## References

1. Cunge, J.A. (1969). On the subject of a flood propagation computation method. *J. Hydraul. Res.*, 7(2), 205-230.
2. Bindas et al. (2024). Improving River Routing Using a Differentiable Muskingum-Cunge Model. *WRR*.
3. Mizukami et al. (2016). mizuRoute version 1. *GMD*, 9(6), 2223-2238.
4. Griewank, A., & Walther, A. (2000). Algorithm 799: Revolve. *ACM TOMS*, 26(1), 19-45.
