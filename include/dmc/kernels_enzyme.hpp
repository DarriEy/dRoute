/**
 * @file kernels_enzyme.hpp
 * @brief Enzyme-Compatible Routing Kernels
 * 
 * Flat-array interface for Muskingum-Cunge routing that can be
 * differentiated by Enzyme's source-to-source transformation.
 * 
 * Design Principles:
 * 1. No AD-specific types - uses plain double throughout
 * 2. Flat array interfaces - no classes or virtual methods
 * 3. AD-safe math - smooth approximations to avoid discontinuities
 * 4. Stateless kernels - all state passed explicitly
 * 5. CUDA-ready - marked with host/device decorators
 * 
 * The flat-array interface mirrors dFUSE's approach for consistency
 * and future coupling.
 */

#ifndef DMC_KERNELS_ENZYME_HPP
#define DMC_KERNELS_ENZYME_HPP

#include "ad_backend.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace dmc {
namespace enzyme {

// ============================================================================
// Constants
// ============================================================================

/// Number of reach properties in flat array
constexpr int NUM_REACH_PROPS = 5;  // length, slope, width_coef, width_exp, manning_n

/// Number of geometry properties
constexpr int NUM_GEOM_PROPS = 4;   // width_coef, width_exp, depth_coef, depth_exp

/// Number of state variables per reach
constexpr int NUM_REACH_STATE = 4;  // Q_in_prev, Q_in_curr, Q_out_prev, lateral

/// Number of auxiliary outputs per reach
constexpr int NUM_AUX_OUT = 5;      // K, X, celerity, C0, C_sum (for debugging)

// ============================================================================
// AD-Safe Math Functions (plain double versions)
// ============================================================================

/**
 * @brief Safe maximum with small epsilon for gradient stability
 */
DMC_HOST_DEVICE inline double safe_max(double a, double b) {
    return (a > b) ? a : b;
}

/**
 * @brief Smooth maximum - differentiable approximation to max(a, b)
 * 
 * Uses: smooth_max(a, b) ≈ 0.5 * (a + b + sqrt((a-b)² + ε))
 */
DMC_HOST_DEVICE inline double smooth_max(double a, double b, double epsilon = 1e-6) {
    double diff = a - b;
    return 0.5 * (a + b + std::sqrt(diff * diff + epsilon));
}

/**
 * @brief Smooth minimum - differentiable approximation to min(a, b)
 */
DMC_HOST_DEVICE inline double smooth_min(double a, double b, double epsilon = 1e-6) {
    double diff = a - b;
    return 0.5 * (a + b - std::sqrt(diff * diff + epsilon));
}

/**
 * @brief Smooth clamp - differentiable approximation to clamp(x, lo, hi)
 */
DMC_HOST_DEVICE inline double smooth_clamp(double x, double lo, double hi, double epsilon = 1e-6) {
    return smooth_min(smooth_max(x, lo, epsilon), hi, epsilon);
}

/**
 * @brief AD-safe power: x^y = exp(y * log(x))
 * 
 * This form ensures gradients flow through both base and exponent.
 * Standard pow(x, y) may not propagate gradients through y correctly.
 */
DMC_HOST_DEVICE inline double ad_safe_pow(double base, double exponent) {
    double safe_base = safe_max(base, 1e-10);
    return std::exp(exponent * std::log(safe_base));
}

// ============================================================================
// Reach Property Accessors
// ============================================================================

/**
 * @brief Pack reach properties into flat array
 * 
 * @param length    Reach length [m]
 * @param slope     Bed slope [m/m]
 * @param manning_n Manning's roughness coefficient
 * @param width_coef Power-law width coefficient
 * @param width_exp  Power-law width exponent
 * @param depth_coef Power-law depth coefficient
 * @param depth_exp  Power-law depth exponent
 * @param props     Output array [7 elements]
 */
DMC_HOST_DEVICE inline void pack_reach_props(
    double length, double slope, double manning_n,
    double width_coef, double width_exp,
    double depth_coef, double depth_exp,
    double* props
) {
    props[0] = length;
    props[1] = slope;
    props[2] = manning_n;
    props[3] = width_coef;
    props[4] = width_exp;
    props[5] = depth_coef;
    props[6] = depth_exp;
}

// Extended property count including depth
constexpr int NUM_REACH_PROPS_FULL = 7;

// ============================================================================
// Hydraulic Computations
// ============================================================================

/**
 * @brief Compute channel width from discharge using power law
 * 
 * W = width_coef * Q^width_exp
 */
DMC_HOST_DEVICE inline double compute_width(double Q, double width_coef, double width_exp) {
    double Q_safe = safe_max(Q, 0.01);
    return width_coef * ad_safe_pow(Q_safe, width_exp);
}

/**
 * @brief Compute channel depth from discharge using power law
 * 
 * D = depth_coef * Q^depth_exp
 */
DMC_HOST_DEVICE inline double compute_depth(double Q, double depth_coef, double depth_exp) {
    double Q_safe = safe_max(Q, 0.01);
    return depth_coef * ad_safe_pow(Q_safe, depth_exp);
}

/**
 * @brief Compute hydraulic radius (rectangular channel approximation)
 * 
 * R = A / P = (W * D) / (W + 2*D)
 */
DMC_HOST_DEVICE inline double compute_hydraulic_radius(double width, double depth) {
    double area = width * depth;
    double perimeter = width + 2.0 * depth;
    return area / safe_max(perimeter, 0.01);
}

/**
 * @brief Compute velocity from Manning's equation
 * 
 * V = (1/n) * R^(2/3) * S^(1/2)
 */
DMC_HOST_DEVICE inline double compute_velocity(double R_h, double slope, double manning_n) {
    double n_safe = safe_max(manning_n, 0.001);
    double s_safe = safe_max(slope, 1e-6);
    return (1.0 / n_safe) * ad_safe_pow(R_h, 2.0/3.0) * std::sqrt(s_safe);
}

/**
 * @brief Compute wave celerity for kinematic wave
 * 
 * c = (5/3) * V  (for wide rectangular channel)
 */
DMC_HOST_DEVICE inline double compute_celerity(double velocity) {
    return (5.0 / 3.0) * velocity;
}

// ============================================================================
// Core Muskingum-Cunge Kernel
// ============================================================================

/**
 * @brief Single-reach Muskingum-Cunge routing kernel (Enzyme-compatible)
 * 
 * This is the core differentiable routing function. It computes the
 * outflow from a reach given inflow, lateral inflow, and reach properties.
 * 
 * The flat-array interface allows Enzyme to differentiate through the
 * entire computation without any AD-specific types.
 * 
 * @param state_in   Input state [Q_in_prev, Q_in_curr, Q_out_prev, lateral_inflow]
 * @param props      Reach properties [length, slope, manning_n, width_coef, width_exp, depth_coef, depth_exp]
 * @param dt         Timestep [s]
 * @param min_flow   Minimum flow threshold [m³/s]
 * @param x_lower    Lower bound for Muskingum X
 * @param x_upper    Upper bound for Muskingum X
 * @param Q_out      Output: discharge at current timestep [m³/s]
 * @param aux_out    Output: auxiliary diagnostics [K, X, celerity, C_sum, velocity]
 */
DMC_HOST_DEVICE inline void muskingum_cunge_kernel(
    const double* state_in,     // [4]: Q_in_prev, Q_in_curr, Q_out_prev, lateral
    const double* props,        // [7]: length, slope, manning_n, width_c, width_e, depth_c, depth_e
    double dt,
    double min_flow,
    double x_lower,
    double x_upper,
    double* Q_out,              // [1]: output discharge
    double* aux_out             // [5]: K, X, celerity, C_sum, velocity
) {
    // Unpack state
    double Q_in_prev = state_in[0];
    double Q_in_curr = state_in[1];
    double Q_out_prev = state_in[2];
    double lateral = state_in[3];
    
    // Unpack properties
    double length = props[0];
    double slope = props[1];
    double manning_n = props[2];
    double width_coef = props[3];
    double width_exp = props[4];
    double depth_coef = props[5];
    double depth_exp = props[6];
    
    // Reference discharge for hydraulic calculations
    // Use smooth max to ensure gradients flow
    double Q_ref = smooth_max(Q_out_prev, min_flow);
    Q_ref = smooth_max(Q_ref, Q_in_curr);
    Q_ref = smooth_max(Q_ref, lateral);
    
    // Compute hydraulics at reference discharge
    double width = compute_width(Q_ref, width_coef, width_exp);
    double depth = compute_depth(Q_ref, depth_coef, depth_exp);
    double R_h = compute_hydraulic_radius(width, depth);
    double velocity = compute_velocity(R_h, slope, manning_n);
    double celerity = compute_celerity(velocity);
    
    // Ensure minimum celerity to prevent division issues
    celerity = smooth_max(celerity, 0.1);
    
    // Muskingum K: travel time through reach
    double K = length / celerity;
    
    // Cap K to reasonable range (at most 10 timesteps)
    double K_max = 10.0 * dt;
    K = smooth_min(K, K_max);
    K = smooth_max(K, dt * 0.1);  // At least 0.1 timesteps
    
    // Muskingum X: weighting factor
    // X = 0.5 - Q / (2 * c * B * S₀ * Δx)
    double X = 0.5 - Q_ref / (2.0 * celerity * width * slope * length);
    X = smooth_clamp(X, x_lower, x_upper);
    
    // Routing coefficients
    double denom = 2.0 * K * (1.0 - X) + dt;
    double C0 = (dt - 2.0 * K * X) / denom;          // Current inflow weight
    double C1 = (dt + 2.0 * K * X) / denom;          // Previous inflow weight
    double C2 = (2.0 * K * (1.0 - X) - dt) / denom;  // Previous outflow weight
    double C3 = 2.0 * dt / denom;                     // Lateral inflow weight
    
    // Muskingum-Cunge equation
    double Q_new = C0 * Q_in_curr + C1 * Q_in_prev + C2 * Q_out_prev + C3 * lateral;
    
    // Ensure non-negative output (smooth for gradient stability)
    *Q_out = smooth_max(Q_new, min_flow);
    
    // Auxiliary outputs for diagnostics
    aux_out[0] = K;
    aux_out[1] = X;
    aux_out[2] = celerity;
    aux_out[3] = C0 + C1 + C2;  // Should sum to ~1 for mass conservation
    aux_out[4] = velocity;
}

/**
 * @brief Simplified kernel without auxiliary outputs
 */
DMC_HOST_DEVICE inline void muskingum_cunge_kernel_simple(
    const double* state_in,
    const double* props,
    double dt,
    double min_flow,
    double* Q_out
) {
    double aux[5];
    muskingum_cunge_kernel(state_in, props, dt, min_flow, 0.0, 0.5, Q_out, aux);
}

// ============================================================================
// Sub-Stepped Routing Kernel
// ============================================================================

/**
 * @brief Muskingum-Cunge kernel with fixed sub-stepping
 * 
 * Uses a fixed number of substeps for numerical stability on short reaches.
 * This is AD-safe because the number of substeps is fixed (no control flow).
 * 
 * @param state_in    Input state [Q_in_prev, Q_in_curr, Q_out_prev, lateral]
 * @param props       Reach properties [7 elements]
 * @param dt          Full timestep [s]
 * @param num_substeps Number of substeps (fixed for AD-safety)
 * @param min_flow    Minimum flow threshold
 * @param x_lower     Lower bound for X
 * @param x_upper     Upper bound for X
 * @param Q_out       Output discharge
 */
DMC_HOST_DEVICE inline void muskingum_cunge_substepped(
    const double* state_in,
    const double* props,
    double dt,
    int num_substeps,
    double min_flow,
    double x_lower,
    double x_upper,
    double* Q_out
) {
    if (num_substeps <= 1) {
        double aux[5];
        muskingum_cunge_kernel(state_in, props, dt, min_flow, x_lower, x_upper, Q_out, aux);
        return;
    }
    
    double Q_in_prev = state_in[0];
    double Q_in_curr = state_in[1];
    double Q_out_prev = state_in[2];
    double lateral = state_in[3];
    
    double sub_dt = dt / num_substeps;
    double dQ_in = (Q_in_curr - Q_in_prev) / num_substeps;
    
    double Q_out_s = Q_out_prev;
    double sub_state[4];
    double aux[5];
    
    for (int s = 0; s < num_substeps; ++s) {
        // Interpolated inflows for this substep
        sub_state[0] = Q_in_prev + dQ_in * s;         // Q_in_prev for substep
        sub_state[1] = Q_in_prev + dQ_in * (s + 1);   // Q_in_curr for substep
        sub_state[2] = Q_out_s;                        // Q_out_prev for substep
        sub_state[3] = lateral;                        // Lateral (constant)
        
        double Q_out_new;
        muskingum_cunge_kernel(sub_state, props, sub_dt, min_flow, 
                               x_lower, x_upper, &Q_out_new, aux);
        Q_out_s = Q_out_new;
    }
    
    *Q_out = Q_out_s;
}

// ============================================================================
// Network Routing Functions
// ============================================================================

/**
 * @brief Route entire network for one timestep
 * 
 * Processes reaches in topological order, gathering upstream flows
 * and applying the MC kernel to each reach.
 * 
 * @param n_reaches       Number of reaches
 * @param topo_order      Reach indices in topological order [n_reaches]
 * @param downstream_idx  Downstream reach index for each reach (-1 for outlet) [n_reaches]
 * @param upstream_counts Number of upstream reaches for each reach [n_reaches]
 * @param upstream_offsets Offset into upstream_indices for each reach [n_reaches]
 * @param upstream_indices Flattened list of upstream reach indices
 * @param reach_props     Reach properties [n_reaches * 7]
 * @param reach_states    Reach states [n_reaches * 4] - MODIFIED
 * @param lateral_inflows Lateral inflows [n_reaches]
 * @param dt              Timestep [s]
 * @param num_substeps    Sub-steps per reach
 * @param min_flow        Minimum flow
 * @param x_lower         Lower bound for X
 * @param x_upper         Upper bound for X
 * @param Q_out           Output discharges [n_reaches]
 */
inline void route_network_timestep(
    int n_reaches,
    const int* topo_order,
    const int* downstream_idx,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,
    double* reach_states,
    const double* lateral_inflows,
    double dt,
    int num_substeps,
    double min_flow,
    double x_lower,
    double x_upper,
    double* Q_out
) {
    // Process reaches in topological order
    for (int i = 0; i < n_reaches; ++i) {
        int reach_id = topo_order[i];
        
        // Gather upstream inflow
        double upstream_inflow = 0.0;
        int n_upstream = upstream_counts[reach_id];
        int offset = upstream_offsets[reach_id];
        
        for (int u = 0; u < n_upstream; ++u) {
            int up_reach = upstream_indices[offset + u];
            upstream_inflow += Q_out[up_reach];
        }
        
        // Current state for this reach
        double* state = &reach_states[reach_id * NUM_REACH_STATE];
        const double* props = &reach_props[reach_id * NUM_REACH_PROPS_FULL];
        
        // Advance inflow state: prev = curr, curr = new upstream
        // NOTE: Lateral inflow is handled SEPARATELY in the MC kernel via C3 term
        // Do NOT add it to upstream_inflow here to avoid double-counting
        state[0] = state[1];                           // Q_in_prev = Q_in_curr
        state[1] = upstream_inflow;                    // Q_in_curr (upstream only)
        // state[2] is Q_out_prev, updated after routing
        state[3] = lateral_inflows[reach_id];          // lateral (handled by kernel)
        
        // Route
        double Q_new;
        muskingum_cunge_substepped(state, props, dt, num_substeps,
                                   min_flow, x_lower, x_upper, &Q_new);
        
        // Store output
        Q_out[reach_id] = Q_new;
        
        // Update state for next timestep
        state[2] = Q_new;  // Q_out_prev = Q_out_curr
    }
}

// ============================================================================
// Loss and Gradient Computation
// ============================================================================

/**
 * @brief Compute MSE loss at gauge locations
 * 
 * @param n_reaches     Number of reaches
 * @param Q_simulated   Simulated discharges [n_reaches]
 * @param n_gauges      Number of gauges
 * @param gauge_reaches Reach indices with gauges [n_gauges]
 * @param Q_observed    Observed discharges [n_gauges]
 * @return MSE loss
 */
DMC_HOST_DEVICE inline double compute_gauge_loss(
    int n_reaches,
    const double* Q_simulated,
    int n_gauges,
    const int* gauge_reaches,
    const double* Q_observed
) {
    double loss = 0.0;
    for (int g = 0; g < n_gauges; ++g) {
        int reach_id = gauge_reaches[g];
        double diff = Q_simulated[reach_id] - Q_observed[g];
        loss += diff * diff;
    }
    return (n_gauges > 0) ? loss / n_gauges : 0.0;
}

/**
 * @brief Multi-timestep simulation with loss computation
 * 
 * This is the main function to differentiate with Enzyme for calibration.
 * 
 * @param n_reaches       Number of reaches
 * @param n_timesteps     Number of timesteps
 * @param topo_order      Topological order [n_reaches]
 * @param downstream_idx  Downstream reach indices [n_reaches]
 * @param upstream_counts Upstream counts [n_reaches]
 * @param upstream_offsets Upstream offsets [n_reaches]
 * @param upstream_indices Upstream indices [sum of upstream_counts]
 * @param reach_props     Reach properties [n_reaches * 7]
 * @param initial_states  Initial states [n_reaches * 4]
 * @param lateral_series  Lateral inflow series [n_timesteps * n_reaches]
 * @param dt              Timestep
 * @param num_substeps    Sub-steps per reach
 * @param min_flow        Minimum flow
 * @param x_lower         Lower X bound
 * @param x_upper         Upper X bound
 * @param n_gauges        Number of gauges
 * @param gauge_reaches   Gauge reach indices [n_gauges]
 * @param observed_series Observed discharge series [n_timesteps * n_gauges]
 * @return Total MSE loss
 */
inline double simulate_and_compute_loss(
    int n_reaches,
    int n_timesteps,
    const int* topo_order,
    const int* downstream_idx,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,
    const double* initial_states,
    const double* lateral_series,
    double dt,
    int num_substeps,
    double min_flow,
    double x_lower,
    double x_upper,
    int n_gauges,
    const int* gauge_reaches,
    const double* observed_series
) {
    // Allocate working arrays
    std::vector<double> reach_states(n_reaches * NUM_REACH_STATE);
    std::vector<double> Q_out(n_reaches);
    
    // Initialize states
    std::memcpy(reach_states.data(), initial_states, 
                n_reaches * NUM_REACH_STATE * sizeof(double));
    
    double total_loss = 0.0;
    
    for (int t = 0; t < n_timesteps; ++t) {
        const double* lateral = &lateral_series[t * n_reaches];
        const double* observed = &observed_series[t * n_gauges];
        
        route_network_timestep(
            n_reaches, topo_order, downstream_idx,
            upstream_counts, upstream_offsets, upstream_indices,
            reach_props, reach_states.data(), lateral,
            dt, num_substeps, min_flow, x_lower, x_upper,
            Q_out.data()
        );
        
        total_loss += compute_gauge_loss(n_reaches, Q_out.data(),
                                         n_gauges, gauge_reaches, observed);
    }
    
    return total_loss / n_timesteps;
}

// ============================================================================
// Enzyme Gradient Wrappers
// ============================================================================

#ifdef DMC_USE_ENZYME

/**
 * @brief Compute gradient of single-reach routing w.r.t. parameters
 * 
 * Uses Enzyme to automatically differentiate the MC kernel.
 */
inline void compute_reach_gradient_enzyme(
    const double* state_in,
    const double* props,
    double dt,
    double min_flow,
    double x_lower,
    double x_upper,
    double dL_dQout,              // Incoming gradient from loss
    double* d_props               // Output: gradient w.r.t. properties [7]
) {
    double Q_out;
    double aux[5];
    double d_state[4] = {0, 0, 0, 0};
    double d_Q_out = dL_dQout;
    double d_aux[5] = {0, 0, 0, 0, 0};
    
    std::memset(d_props, 0, NUM_REACH_PROPS_FULL * sizeof(double));
    
    __enzyme_autodiff(
        (void*)muskingum_cunge_kernel,
        enzyme_const, state_in,
        enzyme_dup, props, d_props,
        enzyme_const, dt,
        enzyme_const, min_flow,
        enzyme_const, x_lower,
        enzyme_const, x_upper,
        enzyme_dup, &Q_out, &d_Q_out,
        enzyme_dupnoneed, aux, d_aux
    );
}

/**
 * @brief Compute gradients of full simulation loss w.r.t. reach properties
 */
inline void compute_simulation_gradient_enzyme(
    int n_reaches,
    int n_timesteps,
    const int* topo_order,
    const int* downstream_idx,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,
    double* d_reach_props,          // Output: gradients [n_reaches * 7]
    const double* initial_states,
    const double* lateral_series,
    double dt,
    int num_substeps,
    double min_flow,
    double x_lower,
    double x_upper,
    int n_gauges,
    const int* gauge_reaches,
    const double* observed_series
) {
    std::memset(d_reach_props, 0, n_reaches * NUM_REACH_PROPS_FULL * sizeof(double));
    
    __enzyme_autodiff(
        (void*)simulate_and_compute_loss,
        enzyme_const, n_reaches,
        enzyme_const, n_timesteps,
        enzyme_const, topo_order,
        enzyme_const, downstream_idx,
        enzyme_const, upstream_counts,
        enzyme_const, upstream_offsets,
        enzyme_const, upstream_indices,
        enzyme_dup, reach_props, d_reach_props,
        enzyme_const, initial_states,
        enzyme_const, lateral_series,
        enzyme_const, dt,
        enzyme_const, num_substeps,
        enzyme_const, min_flow,
        enzyme_const, x_lower,
        enzyme_const, x_upper,
        enzyme_const, n_gauges,
        enzyme_const, gauge_reaches,
        enzyme_const, observed_series
    );
}

#endif // DMC_USE_ENZYME

// ============================================================================
// Numerical Gradient for Validation
// ============================================================================

/**
 * @brief Compute numerical gradient via finite differences
 * 
 * Used to validate Enzyme gradients against a known-correct implementation.
 */
inline void compute_reach_gradient_numerical(
    const double* state_in,
    const double* props,
    double dt,
    double min_flow,
    double x_lower,
    double x_upper,
    double* d_props,
    double eps = 1e-6
) {
    double Q_base, aux[5];
    muskingum_cunge_kernel(state_in, props, dt, min_flow, x_lower, x_upper, &Q_base, aux);
    
    std::vector<double> props_pert(NUM_REACH_PROPS_FULL);
    std::memcpy(props_pert.data(), props, NUM_REACH_PROPS_FULL * sizeof(double));
    
    for (int i = 0; i < NUM_REACH_PROPS_FULL; ++i) {
        double orig = props_pert[i];
        double h = eps * std::max(std::abs(orig), 1.0);
        
        props_pert[i] = orig + h;
        double Q_plus;
        muskingum_cunge_kernel(state_in, props_pert.data(), dt, min_flow, 
                               x_lower, x_upper, &Q_plus, aux);
        
        d_props[i] = (Q_plus - Q_base) / h;
        props_pert[i] = orig;
    }
}

// ============================================================================
// LAG ROUTING KERNEL (Simple Delay)
// ============================================================================

/**
 * @brief Compute lag (delay) in timesteps based on reach properties
 * 
 * lag = travel_time / dt = (length / velocity) / dt
 * velocity = (1/n) * R_h^(2/3) * S^(1/2)
 * 
 * @param length     Reach length [m]
 * @param slope      Bed slope [m/m]
 * @param manning_n  Manning's roughness coefficient
 * @param dt         Timestep [s]
 * @return Lag in timesteps (fractional)
 */
DMC_HOST_DEVICE inline double compute_lag_timesteps(
    double length,
    double slope,
    double manning_n,
    double dt
) {
    double n_safe = safe_max(manning_n, 0.001);
    double s_safe = safe_max(slope, 1e-6);
    double R_h = 1.0;  // Assume unit hydraulic radius for simplicity
    
    double velocity = (1.0 / n_safe) * ad_safe_pow(R_h, 2.0/3.0) * std::sqrt(s_safe);
    velocity = smooth_clamp(velocity, 0.1, 5.0);
    
    double travel_time = length / velocity;
    return travel_time / dt;
}

/**
 * @brief Single-reach lag routing kernel (Enzyme-compatible)
 * 
 * Lag routing delays inflow by a travel-time-based number of timesteps.
 * Uses fractional lag with linear interpolation for AD-friendly behavior.
 * 
 * @param buffer      Inflow history buffer [max_lag elements, oldest first]
 * @param buffer_size Size of buffer
 * @param Q_in_new    New inflow to add to buffer
 * @param lag_frac    Fractional lag in timesteps
 * @param Q_out       Output: lagged discharge
 */
DMC_HOST_DEVICE inline void lag_routing_kernel(
    const double* buffer,
    int buffer_size,
    double Q_in_new,
    double lag_frac,
    double* Q_out
) {
    // Clamp lag to valid range
    lag_frac = smooth_clamp(lag_frac, 0.5, static_cast<double>(buffer_size - 1));
    
    // Integer and fractional parts for interpolation
    int lag_int = static_cast<int>(lag_frac);
    double lag_rem = lag_frac - lag_int;
    
    // Linear interpolation between adjacent buffer positions
    // buffer[0] is newest (just pushed), buffer[buffer_size-1] is oldest
    int idx1 = lag_int;
    int idx2 = lag_int + 1;
    
    if (idx1 >= buffer_size) idx1 = buffer_size - 1;
    if (idx2 >= buffer_size) idx2 = buffer_size - 1;
    
    // Smooth interpolation
    *Q_out = buffer[idx1] * (1.0 - lag_rem) + buffer[idx2] * lag_rem;
}

/**
 * @brief Route entire network using lag method for one timestep
 */
inline void route_network_lag_timestep(
    int n_reaches,
    const int* topo_order,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,      // [n_reaches * 7]
    double* lag_buffers,            // [n_reaches * max_lag]
    int max_lag,
    const double* lateral_inflows,
    double dt,
    double* Q_out
) {
    for (int i = 0; i < n_reaches; ++i) {
        int reach_id = topo_order[i];
        
        // Gather upstream inflow
        double upstream_inflow = 0.0;
        int n_upstream = upstream_counts[reach_id];
        int offset = upstream_offsets[reach_id];
        for (int u = 0; u < n_upstream; ++u) {
            int up_reach = upstream_indices[offset + u];
            upstream_inflow += Q_out[up_reach];
        }
        
        double Q_in = upstream_inflow + lateral_inflows[reach_id];
        
        // Get reach properties
        const double* props = &reach_props[reach_id * NUM_REACH_PROPS_FULL];
        double length = props[0];
        double slope = props[1];
        double manning_n = props[2];
        
        // Compute lag
        double lag_frac = compute_lag_timesteps(length, slope, manning_n, dt);
        
        // Get buffer for this reach
        double* buffer = &lag_buffers[reach_id * max_lag];
        
        // Compute lagged output
        double Q_lagged;
        lag_routing_kernel(buffer, max_lag, Q_in, lag_frac, &Q_lagged);
        
        // Shift buffer and add new inflow
        for (int j = max_lag - 1; j > 0; --j) {
            buffer[j] = buffer[j - 1];
        }
        buffer[0] = Q_in;
        
        Q_out[reach_id] = Q_lagged;
    }
}

// ============================================================================
// IRF (IMPULSE RESPONSE FUNCTION) ROUTING KERNEL
// ============================================================================

/**
 * @brief Gamma PDF (unnormalized) for IRF kernel
 * 
 * f(t; k, θ) ∝ t^(k-1) * exp(-t/θ)
 * 
 * @param t      Time [s]
 * @param k      Shape parameter (typically 2.0-3.0)
 * @param theta  Scale parameter = travel_time / k
 */
DMC_HOST_DEVICE inline double gamma_pdf_unnorm(double t, double k, double theta) {
    if (t <= 0 || theta <= 0) return 0.0;
    return ad_safe_pow(t, k - 1.0) * std::exp(-t / theta);
}

/**
 * @brief Sigmoid function for soft masking
 */
DMC_HOST_DEVICE inline double sigmoid(double x) {
    if (x > 20.0) return 1.0;
    if (x < -20.0) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * @brief Compute IRF kernel weights with soft masking
 * 
 * The kernel is a gamma distribution with soft cutoff:
 *   w[i] = gamma_pdf(t_i) * sigmoid((T_cutoff - t_i) * steepness / scale)
 * 
 * @param kernel_out    Output kernel weights [kernel_size]
 * @param kernel_size   Number of kernel positions
 * @param dt            Timestep [s]
 * @param travel_time   Mean travel time [s]
 * @param shape_k       Gamma shape parameter
 * @param mask_steepness Sigmoid steepness for cutoff
 */
DMC_HOST_DEVICE inline void compute_irf_kernel(
    double* kernel_out,
    int kernel_size,
    double dt,
    double travel_time,
    double shape_k,
    double mask_steepness
) {
    double theta = travel_time / shape_k;  // Scale parameter
    double T_cutoff = 5.0 * theta;         // 99%+ of gamma mass
    
    double sum = 0.0;
    
    for (int i = 0; i < kernel_size; ++i) {
        double t = (i + 0.5) * dt;  // Mid-point of timestep
        
        // Gamma weight
        double w = gamma_pdf_unnorm(t, shape_k, theta);
        
        // Soft mask
        double z = (T_cutoff - t) * mask_steepness / theta;
        double mask = sigmoid(z);
        
        kernel_out[i] = w * mask;
        sum += kernel_out[i];
    }
    
    // Normalize
    if (sum > 1e-10) {
        for (int i = 0; i < kernel_size; ++i) {
            kernel_out[i] /= sum;
        }
    }
}

/**
 * @brief IRF convolution kernel (Enzyme-compatible)
 * 
 * Computes outflow as convolution of inflow history with IRF kernel.
 * 
 * @param inflow_history  Inflow history [kernel_size], newest first
 * @param kernel          IRF kernel weights [kernel_size]
 * @param kernel_size     Size of kernel/history
 * @param Q_out           Output: convolved discharge
 */
DMC_HOST_DEVICE inline void irf_convolution_kernel(
    const double* inflow_history,
    const double* kernel,
    int kernel_size,
    double* Q_out
) {
    double sum = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        sum += inflow_history[i] * kernel[i];
    }
    *Q_out = sum;
}

/**
 * @brief Single-reach IRF routing kernel
 * 
 * @param inflow_history  Inflow history [kernel_size], newest first
 * @param props           Reach properties [7]
 * @param kernel_size     Size of IRF kernel
 * @param dt              Timestep [s]
 * @param shape_k         Gamma shape parameter
 * @param mask_steepness  Mask steepness
 * @param Q_out           Output discharge
 * @param kernel_scratch  Scratch space for kernel computation [kernel_size]
 */
DMC_HOST_DEVICE inline void irf_routing_reach_kernel(
    const double* inflow_history,
    const double* props,
    int kernel_size,
    double dt,
    double shape_k,
    double mask_steepness,
    double* Q_out,
    double* kernel_scratch
) {
    double length = props[0];
    double slope = props[1];
    double manning_n = props[2];
    
    // Compute travel time
    double n_safe = safe_max(manning_n, 0.001);
    double s_safe = safe_max(slope, 1e-6);
    double R_h = 1.0;
    double velocity = (1.0 / n_safe) * ad_safe_pow(R_h, 2.0/3.0) * std::sqrt(s_safe);
    velocity = smooth_clamp(velocity, 0.1, 5.0);
    double travel_time = length / velocity;
    
    // Compute kernel
    compute_irf_kernel(kernel_scratch, kernel_size, dt, travel_time, shape_k, mask_steepness);
    
    // Convolve
    irf_convolution_kernel(inflow_history, kernel_scratch, kernel_size, Q_out);
}

/**
 * @brief Route entire network using IRF method for one timestep
 */
inline void route_network_irf_timestep(
    int n_reaches,
    const int* topo_order,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,
    double* inflow_histories,       // [n_reaches * kernel_size]
    int kernel_size,
    const double* lateral_inflows,
    double dt,
    double shape_k,
    double mask_steepness,
    double* Q_out,
    double* kernel_scratch          // [kernel_size] for each thread
) {
    for (int i = 0; i < n_reaches; ++i) {
        int reach_id = topo_order[i];
        
        // Gather upstream inflow
        double upstream_inflow = 0.0;
        int n_upstream = upstream_counts[reach_id];
        int offset = upstream_offsets[reach_id];
        for (int u = 0; u < n_upstream; ++u) {
            int up_reach = upstream_indices[offset + u];
            upstream_inflow += Q_out[up_reach];
        }
        
        double Q_in = upstream_inflow + lateral_inflows[reach_id];
        
        // Get buffers
        double* history = &inflow_histories[reach_id * kernel_size];
        const double* props = &reach_props[reach_id * NUM_REACH_PROPS_FULL];
        
        // Route using IRF
        double Q_routed;
        irf_routing_reach_kernel(history, props, kernel_size, dt, 
                                 shape_k, mask_steepness, &Q_routed, kernel_scratch);
        
        // Shift history and add new inflow (newest at index 0)
        for (int j = kernel_size - 1; j > 0; --j) {
            history[j] = history[j - 1];
        }
        history[0] = Q_in;
        
        Q_out[reach_id] = Q_routed;
    }
}

// ============================================================================
// KWT-SOFT (KINEMATIC WAVE TRACKING WITH SOFT GATES) KERNEL
// ============================================================================

/// Maximum number of parcels per reach
constexpr int MAX_PARCELS_PER_REACH = 100;

/// Parcel state size
constexpr int PARCEL_STATE_SIZE = 5;  // volume, position, spread, celerity, remaining

/**
 * @brief Compute wave celerity for KWT
 */
DMC_HOST_DEVICE inline double compute_kwt_celerity(
    double Q,
    double manning_n,
    double slope,
    double width_coef,
    double width_exp,
    double depth_coef,
    double depth_exp
) {
    Q = smooth_max(Q, 0.001);
    double n_safe = safe_max(manning_n, 0.001);
    double s_safe = safe_max(slope, 1e-6);
    
    double width = width_coef * ad_safe_pow(Q, width_exp);
    width = safe_max(width, 0.5);
    
    double depth = depth_coef * ad_safe_pow(Q, depth_exp);
    depth = safe_max(depth, 0.05);
    
    double area = width * depth;
    double velocity = Q / area;
    
    // Kinematic wave celerity: c = 5/3 * v
    double celerity = (5.0 / 3.0) * velocity;
    return smooth_clamp(celerity, 0.1, 5.0);
}

/**
 * @brief Soft gate function (smooth step)
 */
DMC_HOST_DEVICE inline double soft_gate(double x, double threshold, double steepness) {
    double z = (x - threshold) * steepness;
    return sigmoid(z);
}

/**
 * @brief Single-reach KWT-soft routing kernel
 * 
 * Tracks wave parcels through the reach with soft-gated exit probabilities.
 * 
 * @param parcel_states   Parcel states [max_parcels * PARCEL_STATE_SIZE]
 * @param n_parcels       Number of active parcels (in/out)
 * @param props           Reach properties [7]
 * @param Q_in            Inflow for this timestep [m³/s]
 * @param dt              Timestep [s]
 * @param gate_steepness  Soft gate steepness
 * @param Q_out           Output: outflow rate [m³/s]
 */
DMC_HOST_DEVICE inline void kwt_soft_reach_kernel(
    double* parcel_states,
    int* n_parcels,
    const double* props,
    double Q_in,
    double dt,
    double gate_steepness,
    double* Q_out
) {
    double length = props[0];
    double slope = props[1];
    double manning_n = props[2];
    double width_coef = props[3];
    double width_exp = props[4];
    double depth_coef = props[5];
    double depth_exp = props[6];
    
    // Create new parcel from inflow
    double inflow_vol = Q_in * dt;
    if (inflow_vol > 1e-6 && *n_parcels < MAX_PARCELS_PER_REACH) {
        double celerity = compute_kwt_celerity(Q_in, manning_n, slope,
                                                width_coef, width_exp,
                                                depth_coef, depth_exp);
        double spread = celerity * dt;
        
        int idx = (*n_parcels) * PARCEL_STATE_SIZE;
        parcel_states[idx + 0] = inflow_vol;      // volume
        parcel_states[idx + 1] = spread / 2.0;    // position (start near upstream)
        parcel_states[idx + 2] = spread;          // spread
        parcel_states[idx + 3] = celerity;        // celerity
        parcel_states[idx + 4] = 1.0;             // remaining fraction
        (*n_parcels)++;
    }
    
    // Advance parcels and compute outflow
    double total_outflow_vol = 0.0;
    int active_count = 0;
    
    for (int p = 0; p < *n_parcels; ++p) {
        int idx = p * PARCEL_STATE_SIZE;
        double volume = parcel_states[idx + 0];
        double position = parcel_states[idx + 1];
        double spread = parcel_states[idx + 2];
        double celerity = parcel_states[idx + 3];
        double remaining = parcel_states[idx + 4];
        
        // Advance position
        position += celerity * dt;
        
        // Spread increases (dispersion)
        spread += 0.1 * celerity * dt;
        
        // Soft-gated exit probability
        double exit_prob = soft_gate(position, length, gate_steepness / spread);
        
        // Volume that exits this timestep
        double prev_exited = 1.0 - remaining;
        double new_exit = smooth_max(exit_prob - prev_exited, 0.0);
        double exit_vol = volume * new_exit;
        
        total_outflow_vol += exit_vol;
        
        // Update remaining
        remaining = smooth_max(1.0 - exit_prob, 0.0);
        
        // Update parcel state
        parcel_states[idx + 1] = position;
        parcel_states[idx + 2] = spread;
        parcel_states[idx + 4] = remaining;
        
        // Keep parcel if still has significant volume remaining
        if (remaining > 0.001) {
            if (active_count != p) {
                // Compact parcels
                int new_idx = active_count * PARCEL_STATE_SIZE;
                for (int k = 0; k < PARCEL_STATE_SIZE; ++k) {
                    parcel_states[new_idx + k] = parcel_states[idx + k];
                }
            }
            active_count++;
        }
    }
    
    *n_parcels = active_count;
    *Q_out = total_outflow_vol / dt;
}

/**
 * @brief Route entire network using KWT-soft method for one timestep
 */
inline void route_network_kwt_timestep(
    int n_reaches,
    const int* topo_order,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,
    double* parcel_states,          // [n_reaches * MAX_PARCELS_PER_REACH * PARCEL_STATE_SIZE]
    int* n_parcels,                 // [n_reaches]
    const double* lateral_inflows,
    double dt,
    double gate_steepness,
    double* Q_out
) {
    for (int i = 0; i < n_reaches; ++i) {
        int reach_id = topo_order[i];
        
        // Gather upstream inflow
        double upstream_inflow = 0.0;
        int n_upstream = upstream_counts[reach_id];
        int offset = upstream_offsets[reach_id];
        for (int u = 0; u < n_upstream; ++u) {
            int up_reach = upstream_indices[offset + u];
            upstream_inflow += Q_out[up_reach];
        }
        
        double Q_in = upstream_inflow + lateral_inflows[reach_id];
        
        // Get parcel states for this reach
        double* parcels = &parcel_states[reach_id * MAX_PARCELS_PER_REACH * PARCEL_STATE_SIZE];
        int* np = &n_parcels[reach_id];
        const double* props = &reach_props[reach_id * NUM_REACH_PROPS_FULL];
        
        // Route
        double Q_routed;
        kwt_soft_reach_kernel(parcels, np, props, Q_in, dt, gate_steepness, &Q_routed);
        
        Q_out[reach_id] = Q_routed;
    }
}

// ============================================================================
// DIFFUSIVE WAVE KERNEL (Crank-Nicolson Implicit)
// ============================================================================

/**
 * @brief Solve tridiagonal system using Thomas algorithm (Enzyme-compatible)
 * 
 * Solves A*x = d where A is tridiagonal with diagonals a, b, c.
 * 
 * @param a     Sub-diagonal [n-1]
 * @param b     Main diagonal [n]
 * @param c     Super-diagonal [n-1]
 * @param d     Right-hand side [n]
 * @param x     Output solution [n]
 * @param n     System size
 * @param work  Scratch space [2*n]
 */
DMC_HOST_DEVICE inline void tridiagonal_solve(
    const double* a,
    const double* b,
    const double* c,
    const double* d,
    double* x,
    int n,
    double* work
) {
    double* c_star = work;
    double* d_star = work + n;
    
    // Forward sweep
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];
    
    for (int i = 1; i < n - 1; ++i) {
        double denom = b[i] - a[i-1] * c_star[i-1];
        c_star[i] = c[i] / denom;
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / denom;
    }
    d_star[n-1] = (d[n-1] - a[n-2] * d_star[n-2]) / (b[n-1] - a[n-2] * c_star[n-2]);
    
    // Back substitution
    x[n-1] = d_star[n-1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_star[i] - c_star[i] * x[i+1];
    }
}

/**
 * @brief Single-reach diffusive wave routing kernel
 * 
 * Uses Crank-Nicolson implicit scheme for stability.
 * 
 * @param Q_nodes       Node discharges [num_nodes], modified in place
 * @param num_nodes     Number of spatial nodes
 * @param props         Reach properties [7]
 * @param Q_upstream    Upstream boundary inflow
 * @param dt            Timestep [s]
 * @param work          Scratch space [5*num_nodes]
 */
DMC_HOST_DEVICE inline void diffusive_wave_reach_kernel(
    double* Q_nodes,
    int num_nodes,
    const double* props,
    double Q_upstream,
    double dt,
    double* work
) {
    double length = props[0];
    double slope = props[1];
    double manning_n = props[2];
    double width_coef = props[3];
    double width_exp = props[4];
    
    double dx = length / (num_nodes - 1);
    
    // Reference flow for linearization
    double Q_ref = safe_max(Q_nodes[num_nodes/2], 0.1);
    
    // Compute hydraulic parameters
    double n_safe = safe_max(manning_n, 0.001);
    double s_safe = safe_max(slope, 1e-6);
    double width = width_coef * ad_safe_pow(Q_ref, width_exp);
    width = safe_max(width, 0.5);
    
    double R_h = 1.0;  // Simplified hydraulic radius
    double velocity = (1.0 / n_safe) * ad_safe_pow(R_h, 2.0/3.0) * std::sqrt(s_safe);
    velocity = smooth_clamp(velocity, 0.1, 5.0);
    
    double celerity = (5.0 / 3.0) * velocity;
    
    // Diffusion coefficient
    double D = Q_ref / (2.0 * width * s_safe);
    D = smooth_clamp(D, 1.0, 10000.0);
    
    // Courant and diffusion numbers
    double Cr = celerity * dt / dx;
    double Df = D * dt / (dx * dx);
    
    // Interior nodes only (excluding boundaries)
    int m = num_nodes - 2;
    if (m <= 0) {
        Q_nodes[num_nodes - 1] = Q_upstream;
        return;
    }
    
    // Allocate diagonals from work array
    double* a_diag = work;
    double* b_diag = work + m;
    double* c_diag = work + 2*m;
    double* d_vec = work + 3*m;
    double* tri_work = work + 4*m;
    
    // Build tridiagonal system (upwind + diffusion)
    for (int i = 0; i < m; ++i) {
        int node = i + 1;
        
        // Main diagonal
        b_diag[i] = 1.0 + Cr + 2.0 * Df;
        
        // Sub-diagonal (upstream)
        if (i > 0) {
            a_diag[i-1] = -(Cr + Df);
        }
        
        // Super-diagonal (downstream diffusion)
        if (i < m - 1) {
            c_diag[i] = -Df;
        }
        
        // RHS
        d_vec[i] = Q_nodes[node];
        
        // Boundary contributions
        if (i == 0) {
            d_vec[i] += (Cr + Df) * Q_upstream;
        }
        if (i == m - 1) {
            // Zero-gradient downstream BC
            b_diag[i] -= Df;
        }
    }
    
    // Handle edge case for small system
    if (m == 1) {
        Q_nodes[1] = d_vec[0] / b_diag[0];
    } else {
        // Solve tridiagonal system
        double* Q_interior = d_vec;  // Reuse for solution
        tridiagonal_solve(a_diag, b_diag, c_diag, d_vec, Q_interior, m, tri_work);
        
        // Update interior nodes
        for (int i = 0; i < m; ++i) {
            Q_nodes[i + 1] = Q_interior[i];
        }
    }
    
    // Set boundary conditions
    Q_nodes[0] = Q_upstream;
    Q_nodes[num_nodes - 1] = Q_nodes[num_nodes - 2];  // Zero-gradient
}

/**
 * @brief Route entire network using diffusive wave method for one timestep
 */
inline void route_network_diffusive_timestep(
    int n_reaches,
    const int* topo_order,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,
    double* reach_Q_nodes,          // [n_reaches * num_nodes]
    int num_nodes,
    const double* lateral_inflows,
    double dt,
    double* Q_out,
    double* work_scratch            // [5 * num_nodes]
) {
    for (int i = 0; i < n_reaches; ++i) {
        int reach_id = topo_order[i];
        
        // Gather upstream inflow
        double upstream_inflow = 0.0;
        int n_upstream = upstream_counts[reach_id];
        int offset = upstream_offsets[reach_id];
        for (int u = 0; u < n_upstream; ++u) {
            int up_reach = upstream_indices[offset + u];
            upstream_inflow += Q_out[up_reach];
        }
        
        double Q_in = upstream_inflow + lateral_inflows[reach_id];
        
        // Get node array for this reach
        double* Q_nodes = &reach_Q_nodes[reach_id * num_nodes];
        const double* props = &reach_props[reach_id * NUM_REACH_PROPS_FULL];
        
        // Route using diffusive wave
        diffusive_wave_reach_kernel(Q_nodes, num_nodes, props, Q_in, dt, work_scratch);
        
        // Output is discharge at downstream end
        Q_out[reach_id] = Q_nodes[num_nodes - 1];
    }
}

} // namespace enzyme
} // namespace dmc

#endif // DMC_KERNELS_ENZYME_HPP
