/**
 * @file router_fixes.hpp
 * @brief Documentation of bugs found and fixes for dMC-Route
 * 
 * This file documents the issues found in the routing implementations
 * and provides the corrected code snippets to apply.
 */

#ifndef DMC_ROUTER_FIXES_HPP
#define DMC_ROUTER_FIXES_HPP

/*
=============================================================================
BUG ANALYSIS AND FIXES FOR dMC-Route
=============================================================================

Test Results Analysis:
- MC mass balance: 56.25% error
- DW mass balance: 82.56% error  
- IRF gradient mismatch: 27% to 2500% error between AD and FD
- DW gradient: OPPOSITE SIGN between AD and FD (AD: -15.8, FD: +57.7)
- Diffusive-IFT: Appears frozen at 0% (excessive sub-stepping)

=============================================================================
BUG 1: Diffusive Wave Source Term (causes 82% mass loss)
=============================================================================

Location: router.hpp, DiffusiveWaveRouter::route_reach_diffusive(), lines 1224-1256

PROBLEM:
1. Source term scaling is incorrect: uses q_source * sub_dt / dt
2. Boundary condition incorrectly adds q_source to upstream BC
3. Missing wave celerity factor in source term

The diffusive wave PDE is:
  ∂Q/∂t + c·∂Q/∂x = D·∂²Q/∂x² + c·q_lat

After discretization, the source term per substep should be:
  ΔQ_source = c * q_lat * sub_dt  (at each node)

Current code does:
  source = (lateral_inflow * dx / length) * sub_dt / dt  // WRONG!

FIX - Replace lines 1224-1256 with:
*/

// FIXED SOURCE TERM AND BOUNDARY CONDITIONS:
/*
    // Distribute lateral inflow across interior nodes (not boundary)
    // q_lat has units m²/s (m³/s per m of reach length)
    Real q_lat = reach.lateral_inflow / Real(reach.length);
    
    // Sub-stepping loop
    for (int s = 0; s < sub_steps; ++s) {
        std::vector<Real> Q_old = Q;
        
        // Update interior nodes
        for (int i = 1; i < n_nodes - 1; ++i) {
            // Upwind advection
            Real advection = -Cr * (Q_old[i] - Q_old[i-1]);
            
            // Central diffusion
            Real diffusion = Df * (Q_old[i+1] - Real(2.0) * Q_old[i] + Q_old[i-1]);
            
            // Source term: c * q_lat * sub_dt
            // This is the physically correct discretization of the wave equation
            Real source = c * q_lat * sub_dt;
            
            // Update
            Q[i] = Q_old[i] + advection + diffusion + source;
            
            // Ensure non-negative
            if (config_.enable_gradients && config_.use_smooth_bounds) {
                Q[i] = smooth_max(Q[i], Real(0.0), config_.smooth_epsilon);
            } else if (to_double(Q[i]) < 0.0) {
                Q[i] = Real(0.0);
            }
        }
        
        // Boundary conditions - NO lateral inflow at boundary!
        Q[0] = Q_upstream;  // Fixed: removed + q_source
        Q[n_nodes-1] = Q[n_nodes-2];  // Zero gradient
    }
*/

/*
=============================================================================
BUG 2: Diffusive Wave IFT Excessive Sub-stepping (causes freeze at 0%)
=============================================================================

Location: advanced_routing.hpp, DiffusiveWaveIFT::route_timestep(), lines 372-375

PROBLEM:
The stability check can result in thousands of substeps for high Courant numbers:
  int num_substeps = std::max(num_substeps_cr, num_substeps_df);

With no upper bound, if Cr = 150 (common for large Q and small dx), we get 100+ substeps.
At 26303 timesteps × 49 reaches × 100 substeps = 129M iterations!

FIX - Add maximum substep limit:
*/

// Replace lines 372-375 in advanced_routing.hpp:
/*
    int num_substeps_cr = std::max(1, static_cast<int>(std::ceil(Cr / 1.5)));
    int num_substeps_df = std::max(1, static_cast<int>(std::ceil(Df / 0.8)));
    int num_substeps = std::min(20, std::max(num_substeps_cr, num_substeps_df));  // CAP AT 20!
    
    // If we need more than 20 substeps, reduce spatial resolution instead
    if (num_substeps == 20) {
        // Warn once per simulation
        static bool warned = false;
        if (!warned) {
            std::cerr << "Warning: Capping substeps at 20, consider reducing nodes_per_reach\n";
            warned = true;
        }
    }
*/

/*
=============================================================================
BUG 3: IRF Gradient Mismatch (27% to 2500% error)
=============================================================================

Location: router.hpp, IRFRouter::route_reach_irf(), lines 811-831

PROBLEM:
The analytical gradient formula doesn't match the actual soft-masked computation:
- AD computes gradient through the soft-masked convolution
- Analytical gradient uses the un-masked gamma formula

The mask changes the effective kernel weights, so dQ/dn is different.

FIX - Use AD gradients only, remove analytical override:
*/

// In IRFRouter::compute_gradients(), use tape-based gradients instead of analytical:
/*
inline void IRFRouter::compute_gradients(const std::vector<int>& gauge_reaches,
                                          const std::vector<double>& dL_dQ) {
    if (!AD_ENABLED || gauge_reaches.empty() || dL_dQ.empty()) return;
    
    // Register outputs and seed
    for (size_t i = 0; i < gauge_reaches.size(); ++i) {
        Reach& reach = network_.get_reach(gauge_reaches[i]);
        register_output(reach.outflow_curr);
        set_gradient(reach.outflow_curr, dL_dQ[i]);
    }
    
    // Reverse pass
    evaluate_tape();
    
    // Collect gradients from tape
    network_.collect_gradients();
    
    deactivate_tape();
}
*/

// AND need to enable tape recording in IRFRouter:
/*
inline void IRFRouter::start_recording() {
    if (!config_.enable_gradients || !AD_ENABLED) return;
    
    activate_tape();
    recording_ = true;
    
    // Register manning_n as inputs for each reach
    for (int reach_id : network_.topological_order()) {
        Reach& reach = network_.get_reach(reach_id);
        register_input(reach.manning_n);
    }
}
*/

/*
=============================================================================
BUG 4: Muskingum-Cunge Mass Balance (56% error in test)
=============================================================================

Location: router.hpp, MuskingumCungeRouter::route_reach(), lines 354-392

PROBLEM:
The sub-stepping with C4 lateral term may have an accumulation issue.
Each substep adds C4 * lateral, where C4 is computed for sub_dt.
Over N substeps, total lateral = N * C4(sub_dt) * lateral.

But the comment says "N * C4(sub_dt) * lateral ≈ C4(dt) * lateral".
This approximation may not hold for all parameter ranges.

Let's check: 
  C4(dt) = 2 * dt / (2K(1-X) + dt)
  C4(sub_dt) = 2 * sub_dt / (2K(1-X) + sub_dt)
  
For sub_dt = dt/N:
  C4(sub_dt) = 2*(dt/N) / (2K(1-X) + dt/N)
  
  N * C4(sub_dt) = 2*dt / (2K(1-X) + dt/N)
                 ≠ C4(dt) = 2*dt / (2K(1-X) + dt)
                 
The denominators differ! When K is small relative to dt, this causes significant error.

FIX - Add lateral inflow only once per full timestep, not per substep:
*/

// Replace the sub-stepping loop in route_reach():
/*
    if (config_.fixed_substepping && config_.num_substeps > 1) {
        double sub_dt = dt / config_.num_substeps;
        
        Real Q_in_prev = reach.inflow_prev;
        Real Q_in_curr = reach.inflow_curr;
        Real Q_out_prev = reach.outflow_prev;
        Real lateral = reach.lateral_inflow;
        
        Real dQ_in = (Q_in_curr - Q_in_prev) / Real(config_.num_substeps);
        
        Real Q_out = Q_out_prev;
        for (int s = 0; s < config_.num_substeps; ++s) {
            Real Q_in_s_prev = Q_in_prev + dQ_in * Real(s);
            Real Q_in_s_curr = Q_in_prev + dQ_in * Real(s + 1);
            
            Real C1, C2, C3, C4;
            compute_routing_coefficients(K, X, sub_dt, C1, C2, C3, C4);
            
            // FIXED: Only add lateral on first substep (scaled for full dt)
            Real C4_full;
            if (s == 0) {
                Real denom_full = 2.0 * K * (1.0 - X) + dt;
                C4_full = 2.0 * dt / denom_full;
            } else {
                C4_full = Real(0.0);
            }
            
            Real Q_out_new = C1 * Q_in_s_curr + 
                             C2 * Q_in_s_prev + 
                             C3 * Q_out +
                             C4_full * lateral;  // Use full-dt coefficient
            
            Q_out = safe_max(Q_out_new, Real(config_.min_flow));
        }
        
        return Q_out;
    }
*/

/*
=============================================================================
BUG 5: IRF Performance (slow due to per-timestep kernel recomputation)
=============================================================================

Location: router.hpp, IRFRouter::route_reach_irf(), lines 794-808

PROBLEM:
For each reach at each timestep, we compute max_kernel_size masked weights
with exp() and pow() operations. With 200 kernel positions × 49 reaches × 26303 
timesteps = 257 million exp() calls!

FIX - Cache the masked kernel for each reach, only recompute when manning_n changes:
*/

// Add to IRFRouter class members:
/*
    // Cached masked kernels (only recompute when parameters change)
    std::unordered_map<int, std::vector<Real>> cached_masked_kernels_;
    std::unordered_map<int, Real> cached_manning_n_;
    
    void update_cached_kernel_if_needed(int reach_id, const Real& manning_n, const Real& travel_time);
*/

// Implementation:
/*
inline void IRFRouter::update_cached_kernel_if_needed(int reach_id, const Real& manning_n, 
                                                       const Real& travel_time) {
    // Check if manning_n has changed significantly
    if (cached_manning_n_.count(reach_id)) {
        double diff = std::abs(to_double(manning_n) - to_double(cached_manning_n_[reach_id]));
        if (diff < 1e-8) return;  // No update needed
    }
    
    // Recompute masked kernel
    std::vector<Real> kernel(max_kernel_size_);
    Real weight_sum = Real(0.0);
    
    for (int i = 0; i < max_kernel_size_; ++i) {
        kernel[i] = compute_masked_weight(reach_id, i, travel_time);
        weight_sum = weight_sum + kernel[i];
    }
    
    // Pre-normalize
    if (to_double(weight_sum) > 1e-10) {
        for (int i = 0; i < max_kernel_size_; ++i) {
            kernel[i] = kernel[i] / weight_sum;
        }
    }
    
    cached_masked_kernels_[reach_id] = kernel;
    cached_manning_n_[reach_id] = manning_n;
}

// Then in route_reach_irf, use the cached kernel:
inline void IRFRouter::route_reach_irf(Reach& reach) {
    // ... compute travel_time ...
    
    // Use cached kernel
    update_cached_kernel_if_needed(reach.id, reach.manning_n, travel_time);
    const auto& kernel = cached_masked_kernels_[reach.id];
    
    // Convolution with cached kernel (no exp/pow per timestep!)
    Real Q_out = Real(0.0);
    for (int i = 0; i < n_hist && i < max_kernel_size_; ++i) {
        Q_out = Q_out + history[i] * kernel[i];
    }
    // ... rest of function ...
}
*/

/*
=============================================================================
BUG 6: KWT-Soft Gradient Magnitude (10^5 vs 10^-2 for other methods)
=============================================================================

Location: advanced_routing.hpp, SoftGatedKWT::route_timestep(), lines 841-843

PROBLEM:
Gradients are accumulated across all timesteps without normalization:
  grad_manning_n_[reach_id] += timestep_grad;

With 26303 timesteps, the accumulated gradient is 26303× larger than 
a per-timestep gradient.

FIX - Either normalize by number of timesteps, or use time-average:
*/

// Option 1: Normalize in compute_gradients:
/*
inline void SoftGatedKWT::compute_gradients(const std::vector<int>& gauge_reaches,
                                             const std::vector<double>& dL_dQ) {
    // Normalize gradients by number of accumulated timesteps
    int num_timesteps = static_cast<int>(current_time_ / config_.dt);
    if (num_timesteps > 0) {
        for (auto& [id, g] : grad_manning_n_) {
            network_.get_reach(id).grad_manning_n = g / num_timesteps;
        }
    }
}
*/

// Option 2: More physically meaningful - use final timestep gradient only:
/*
inline void SoftGatedKWT::route_timestep() {
    // ... existing code ...
    
    // Replace accumulation with assignment (gradient of final state)
    if (recording_ && enable_gradients_) {
        grad_manning_n_[reach_id] = timestep_grad;  // = not +=
    }
}
*/

/*
=============================================================================
SUMMARY OF FIXES TO APPLY:
=============================================================================

1. router.hpp line 1224-1256: Fix DW source term and boundary condition
2. advanced_routing.hpp line 372-375: Cap DW-IFT substeps at 20
3. router.hpp lines 866-923: Use AD tape for IRF gradients instead of analytical
4. router.hpp lines 354-392: Fix MC lateral inflow in sub-stepping
5. router.hpp: Cache IRF kernels to avoid repeated exp/pow computation
6. advanced_routing.hpp line 842: Normalize KWT-Soft gradients by timestep count

PERFORMANCE PRIORITIES:
- DW-IFT freeze: Fix #2 (immediate impact, seconds vs hours)
- IRF slowness: Fix #5 (10-50× speedup expected)
- MC slowness: Already reasonable, just fix correctness with #4

CORRECTNESS PRIORITIES:
- DW 82% mass loss: Fix #1 (critical for usability)
- IRF gradient mismatch: Fix #3 (critical for calibration)
- MC 56% mass imbalance: Fix #4 (affects calibration quality)
- KWT-Soft gradient scale: Fix #6 (affects optimizer step sizes)

*/

#endif // DMC_ROUTER_FIXES_HPP
