/**
 * Gradient verification via finite differences.
 * 
 * Compares AD-computed gradients against finite difference approximations
 * to verify correctness of the automatic differentiation implementation.
 */

#include <dmc/types.hpp>
#include <dmc/network.hpp>
#include <dmc/router.hpp>
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace dmc;

/**
 * Create a simple test network: 3 reaches in series.
 */
Network create_test_network() {
    Network net;
    
    // Three reaches in series
    for (int i = 0; i < 3; ++i) {
        Reach r;
        r.id = i;
        r.name = "reach_" + std::to_string(i);
        r.length = 5000.0;  // 5 km
        r.slope = 0.001;
        r.manning_n = Real(0.035);
        r.upstream_junction_id = (i > 0) ? i - 1 : -1;
        r.downstream_junction_id = i;
        net.add_reach(r);
        
        Junction j;
        j.id = i;
        j.upstream_reach_ids = {i};  // This reach flows INTO this junction
        if (i < 2) j.downstream_reach_ids = {i + 1};  // Next reach flows OUT
        j.is_headwater = (i == 0);
        j.is_outlet = (i == 2);
        net.add_junction(j);
    }
    
    net.build_topology();
    return net;
}

/**
 * Reset network state to initial conditions.
 */
void reset_network_state(Network& net) {
    for (int reach_id : net.topological_order()) {
        Reach& reach = net.get_reach(reach_id);
        reach.inflow_prev = Real(0.0);
        reach.inflow_curr = Real(0.0);
        reach.outflow_prev = Real(0.0);
        reach.outflow_curr = Real(0.0);
        reach.lateral_inflow = Real(0.0);
    }
}

/**
 * Run simulation and return outlet discharge.
 */
double run_simulation(Network& net, int num_steps, double dt) {
    // Reset state before each run
    reset_network_state(net);
    
    RouterConfig config;
    config.dt = dt;
    config.enable_gradients = false;
    
    MuskingumCungeRouter router(net, config);
    
    // Set constant lateral inflow
    for (int reach_id : net.topological_order()) {
        router.set_lateral_inflow(reach_id, 1.0);  // 1 m³/s per reach
    }
    
    // Run simulation
    for (int t = 0; t < num_steps; ++t) {
        router.route_timestep();
    }
    
    // Return outlet discharge (last reach)
    int outlet_id = net.topological_order().back();
    return router.get_discharge(outlet_id);
}

/**
 * Compute gradient via finite differences.
 */
double finite_difference_gradient(Network& net, int reach_id, 
                                   const std::string& param_name,
                                   int num_steps, double dt, double eps = 1e-5) {
    Reach& reach = net.get_reach(reach_id);
    
    // Get pointer to parameter
    Real* param = nullptr;
    if (param_name == "manning_n") {
        param = &reach.manning_n;
    } else if (param_name == "width_coef") {
        param = &reach.geometry.width_coef;
    } else {
        throw std::runtime_error("Unknown parameter: " + param_name);
    }
    
    // Save original value
    double orig = to_double(*param);
    
    // Forward perturbation
    *param = Real(orig + eps);
    double Q_plus = run_simulation(net, num_steps, dt);
    
    // Backward perturbation
    *param = Real(orig - eps);
    double Q_minus = run_simulation(net, num_steps, dt);
    
    // Restore original
    *param = Real(orig);
    
    // Central difference
    return (Q_plus - Q_minus) / (2.0 * eps);
}

/**
 * Compute gradient via automatic differentiation.
 */
double ad_gradient(Network& net, int reach_id, const std::string& param_name,
                   int num_steps, double dt) {
    if (!AD_ENABLED) {
        std::cerr << "AD not enabled, cannot compute AD gradient\n";
        return 0.0;
    }
    
    // Reset state before AD run (critical for matching FD!)
    reset_network_state(net);
    
    RouterConfig config;
    config.dt = dt;
    config.enable_gradients = true;
    
    MuskingumCungeRouter router(net, config);
    
    // Reset any previous AD tape state
    router.reset_gradients();
    
    // Set constant lateral inflow
    for (int rid : net.topological_order()) {
        router.set_lateral_inflow(rid, 1.0);
    }
    
    // Start recording
    router.start_recording();
    
    // Run simulation
    for (int t = 0; t < num_steps; ++t) {
        router.route_timestep();
    }
    
    // Stop recording
    router.stop_recording();
    
    // Compute gradients (dL/dQ = 1 for single output)
    int outlet_id = net.topological_order().back();
    std::vector<int> gauges = {outlet_id};
    std::vector<double> dL_dQ = {1.0};
    router.compute_gradients(gauges, dL_dQ);
    
    // Get gradient
    auto grads = router.get_gradients();
    std::string key = "reach_" + std::to_string(reach_id) + "_" + param_name;
    
    double grad = grads.count(key) ? grads.at(key) : 0.0;
    
    // Clean up
    router.reset_gradients();
    
    return grad;
}

int main() {
    std::cout << "=== Gradient Verification Test ===\n\n";
    
    if (!AD_ENABLED) {
        std::cout << "WARNING: AD is not enabled. Only finite difference test will run.\n";
    }
    
    Network net = create_test_network();
    
    int num_steps = 50;  // Increased to allow flow propagation through network
    double dt = 3600.0;  // 1 hour
    
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Configuration:\n";
    std::cout << "  Timesteps: " << num_steps << "\n";
    std::cout << "  dt: " << dt << " s\n";
    std::cout << "  Reaches: " << net.num_reaches() << "\n\n";
    
    // Test parameters
    std::vector<std::string> params = {"manning_n", "width_coef"};
    
    bool all_passed = true;
    double tol = 0.05;  // 5% relative tolerance (realistic for FD vs AD comparison)
    
    for (int reach_id : net.topological_order()) {
        for (const auto& param : params) {
            double fd_grad = finite_difference_gradient(net, reach_id, param, 
                                                         num_steps, dt);
            double ad_grad = 0.0;
            
            if (AD_ENABLED) {
                ad_grad = ad_gradient(net, reach_id, param, num_steps, dt);
            }
            
            double rel_error = 0.0;
            double abs_diff = std::abs(ad_grad - fd_grad);
            
            if (std::abs(fd_grad) > 1e-6) {
                rel_error = abs_diff / std::abs(fd_grad);
            } else if (std::abs(ad_grad) > 1e-6) {
                rel_error = 1.0;  // FD is zero but AD is not
            }
            // If both are very small, consider it a pass
            
            // Pass if: relative error < tol OR both gradients are negligibly small
            bool passed = (rel_error < tol) || (abs_diff < 1e-6) || !AD_ENABLED;
            
            std::cout << "Reach " << reach_id << ", " << param << ":\n";
            std::cout << "  FD gradient:  " << fd_grad << "\n";
            if (AD_ENABLED) {
                std::cout << "  AD gradient:  " << ad_grad << "\n";
                std::cout << "  Rel. error:   " << rel_error << "\n";
                std::cout << "  Status:       " << (passed ? "PASS" : "FAIL") << "\n";
            }
            std::cout << "\n";
            
            if (!passed) all_passed = false;
        }
    }
    
    if (AD_ENABLED) {
        std::cout << "=== Overall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") 
                  << " ===\n";
        return all_passed ? 0 : 1;
    } else {
        std::cout << "=== FD-only test complete (AD not enabled) ===\n";
        return 0;
    }
}
