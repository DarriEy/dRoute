/**
 * @file parallel_routing.hpp
 * @brief Parallel routing utilities using OpenMP and graph coloring
 * 
 * Provides parallelization for large river networks by identifying
 * independent reaches that can be processed concurrently.
 */

#ifndef DMC_PARALLEL_ROUTING_HPP
#define DMC_PARALLEL_ROUTING_HPP

#include "kernels_enzyme.hpp"
#include "advanced_routing.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dmc {
namespace enzyme {

/**
 * Parallel network routing using graph coloring.
 * 
 * Reaches within the same color group have no dependencies on each other
 * and can be processed in parallel. Color groups must still be processed
 * in dependency order (topological order across groups).
 * 
 * For a typical dendritic network:
 * - Color 0: All headwater reaches (can run in parallel)
 * - Color 1: Reaches directly downstream of headwaters
 * - etc.
 * 
 * This can provide significant speedup for large basins.
 */
inline void route_network_timestep_parallel(
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
    double* Q_out,
    const std::vector<std::vector<int>>& color_groups,
    int num_threads = 0
) {
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif
    
    // Process each color group sequentially (dependencies between groups)
    for (const auto& group : color_groups) {
        // Within a color group, reaches are independent - parallelize
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < group.size(); ++i) {
            int reach_id = group[i];
            
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
            
            // Advance inflow state
            state[0] = state[1];                           // Q_in_prev = Q_in_curr
            state[1] = upstream_inflow;                    // Q_in_curr
            state[3] = lateral_inflows[reach_id];          // lateral
            
            // Route
            double Q_new;
            muskingum_cunge_substepped(state, props, dt, num_substeps,
                                       min_flow, x_lower, x_upper, &Q_new);
            
            // Store output
            Q_out[reach_id] = Q_new;
            state[2] = Q_new;  // Q_out_prev = Q_out_curr
        }
    }
}

/**
 * Parallel routing for any method using graph coloring.
 */
inline void route_network_timestep_method_parallel(
    int method,
    int n_reaches,
    const int* topo_order,
    const int* downstream_idx,
    const int* upstream_counts,
    const int* upstream_offsets,
    const int* upstream_indices,
    const double* reach_props,
    double* reach_states,
    double* extended_state,
    const double* lateral_inflows,
    double dt,
    int num_substeps,
    double min_flow,
    double gate_steepness,
    double* Q_out,
    const std::vector<std::vector<int>>& color_groups,
    int num_threads = 0
) {
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif
    
    // Get extended state size
    int ext_state_size = 0;
    switch (method) {
        case 1: ext_state_size = LAG_STATE_SIZE; break;
        case 2: ext_state_size = IRF_STATE_SIZE; break;
        case 3: ext_state_size = KWT_STATE_SIZE; break;
        case 4: ext_state_size = DW_STATE_SIZE; break;
        default: ext_state_size = 0;
    }
    
    // Process each color group sequentially
    for (const auto& group : color_groups) {
        // Within a color group, parallelize
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (size_t i = 0; i < group.size(); ++i) {
            int reach_id = group[i];
            
            // Gather upstream inflow
            double upstream_inflow = 0.0;
            int n_upstream = upstream_counts[reach_id];
            int offset = upstream_offsets[reach_id];
            
            for (int u = 0; u < n_upstream; ++u) {
                int up_reach = upstream_indices[offset + u];
                upstream_inflow += Q_out[up_reach];
            }
            
            // Add lateral inflow
            double total_inflow = upstream_inflow + lateral_inflows[reach_id];
            
            // Current state for this reach
            double* state = &reach_states[reach_id * NUM_REACH_STATE];
            const double* props = &reach_props[reach_id * NUM_REACH_PROPS_FULL];
            double* ext_state = (ext_state_size > 0) 
                ? &extended_state[reach_id * ext_state_size] 
                : nullptr;
            
            // Advance inflow state
            state[0] = state[1];
            state[1] = total_inflow;
            state[3] = lateral_inflows[reach_id];
            
            // Route based on method
            double Q_new;
            switch (method) {
                case 0: // MC
                    muskingum_cunge_substepped(state, props, dt, num_substeps,
                                               min_flow, 0.0, 0.5, &Q_new);
                    break;
                case 1: // Lag
                    lag_kernel(ext_state, props, total_inflow, dt, &Q_new);
                    break;
                case 2: // IRF
                    irf_kernel(ext_state, props, total_inflow, dt, &Q_new);
                    break;
                case 3: // KWT
                    kwt_kernel(ext_state, props, total_inflow, 
                               dt, gate_steepness, &Q_new);
                    break;
                case 4: // Diffusive
                    diffusive_kernel(ext_state, props, upstream_inflow, 
                                     lateral_inflows[reach_id], dt, num_substeps, &Q_new);
                    break;
                default:
                    Q_new = total_inflow;
            }
            
            // Store output
            Q_out[reach_id] = Q_new;
            state[2] = Q_new;
        }
    }
}

} // namespace enzyme

// ============================================================================
// ParallelEnzymeRouter - Wrapper with OpenMP support
// ============================================================================

/**
 * Thread-safe router using graph coloring for parallelization.
 */
class ParallelEnzymeRouter {
public:
    ParallelEnzymeRouter(Network& network, 
                         int num_threads = 0,
                         double dt = 3600.0,
                         int num_substeps = 4)
        : network_(network)
        , coloring_(network)
        , dt_(dt)
        , num_substeps_(num_substeps)
        , num_threads_(num_threads)
    {
        network_.build_topology();
        initialize_arrays();
        
#ifdef _OPENMP
        if (num_threads_ <= 0) {
            num_threads_ = omp_get_max_threads();
        }
#else
        num_threads_ = 1;
#endif
    }
    
    void route_timestep() {
        // Gather lateral inflows
        for (int reach_id : topo_order_) {
            lateral_inflows_[reach_id] = to_double(network_.get_reach(reach_id).lateral_inflow);
        }
        
        // Route using parallel kernel
        enzyme::route_network_timestep_parallel(
            n_reaches_,
            topo_order_.data(),
            downstream_idx_.data(),
            upstream_counts_.data(),
            upstream_offsets_.data(),
            upstream_indices_.data(),
            reach_props_.data(),
            reach_states_.data(),
            lateral_inflows_.data(),
            dt_,
            num_substeps_,
            1e-10,  // min_flow
            0.0,    // x_lower
            0.5,    // x_upper
            Q_out_.data(),
            coloring_.get_color_groups(),
            num_threads_
        );
        
        // Write back
        for (int reach_id : topo_order_) {
            Reach& reach = network_.get_reach(reach_id);
            reach.outflow_curr = Real(Q_out_[reach_id]);
        }
    }
    
    void set_lateral_inflow(int reach_id, double inflow) {
        network_.get_reach(reach_id).lateral_inflow = Real(inflow);
    }
    
    double get_discharge(int reach_id) const {
        return Q_out_[reach_id];
    }
    
    void reset_state() {
        std::fill(reach_states_.begin(), reach_states_.end(), 0.0);
        std::fill(Q_out_.begin(), Q_out_.end(), 0.0);
    }
    
    int num_colors() const { return coloring_.num_colors(); }
    int num_threads() const { return num_threads_; }
    
    // Zero-copy access to output array
    double* get_discharge_ptr() { return Q_out_.data(); }
    const double* get_discharge_ptr() const { return Q_out_.data(); }
    
private:
    Network& network_;
    GraphColoring coloring_;
    double dt_;
    int num_substeps_;
    int num_threads_;
    
    int n_reaches_ = 0;
    std::vector<int> topo_order_;
    std::vector<int> downstream_idx_;
    std::vector<int> upstream_counts_;
    std::vector<int> upstream_offsets_;
    std::vector<int> upstream_indices_;
    
    std::vector<double> reach_props_;
    std::vector<double> reach_states_;
    std::vector<double> lateral_inflows_;
    std::vector<double> Q_out_;
    
    void initialize_arrays() {
        topo_order_ = network_.topological_order();
        n_reaches_ = static_cast<int>(topo_order_.size());
        
        std::unordered_map<int, int> reach_to_idx;
        for (int i = 0; i < n_reaches_; ++i) {
            reach_to_idx[topo_order_[i]] = i;
        }
        
        downstream_idx_.resize(n_reaches_, -1);
        upstream_counts_.resize(n_reaches_, 0);
        upstream_offsets_.resize(n_reaches_, 0);
        
        reach_props_.resize(n_reaches_ * enzyme::NUM_REACH_PROPS_FULL);
        reach_states_.resize(n_reaches_ * enzyme::NUM_REACH_STATE, 0.0);
        lateral_inflows_.resize(n_reaches_, 0.0);
        Q_out_.resize(n_reaches_, 0.0);
        
        std::vector<std::vector<int>> upstream_lists(n_reaches_);
        
        for (int reach_id : topo_order_) {
            const Reach& reach = network_.get_reach(reach_id);
            
            int idx = reach_id;
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 0] = reach.length;
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 1] = reach.slope;
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 2] = to_double(reach.manning_n);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 3] = to_double(reach.geometry.width_coef);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 4] = to_double(reach.geometry.width_exp);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 5] = to_double(reach.geometry.depth_coef);
            reach_props_[idx * enzyme::NUM_REACH_PROPS_FULL + 6] = to_double(reach.geometry.depth_exp);
            
            if (reach.downstream_junction_id >= 0) {
                try {
                    const Junction& junc = network_.get_junction(reach.downstream_junction_id);
                    for (int down_id : junc.downstream_reach_ids) {
                        if (reach_to_idx.count(down_id)) {
                            downstream_idx_[reach_id] = down_id;
                            upstream_lists[down_id].push_back(reach_id);
                            break;  // Only first downstream
                        }
                    }
                } catch (...) {}
            }
        }
        
        int total_upstream = 0;
        for (int i = 0; i < n_reaches_; ++i) {
            upstream_offsets_[i] = total_upstream;
            upstream_counts_[i] = static_cast<int>(upstream_lists[i].size());
            total_upstream += upstream_counts_[i];
        }
        
        upstream_indices_.resize(total_upstream);
        for (int i = 0; i < n_reaches_; ++i) {
            int offset = upstream_offsets_[i];
            for (size_t j = 0; j < upstream_lists[i].size(); ++j) {
                upstream_indices_[offset + j] = upstream_lists[i][j];
            }
        }
    }
};

} // namespace dmc

#endif // DMC_PARALLEL_ROUTING_HPP
