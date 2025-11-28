/**
 * Single reach routing test.
 * 
 * Verifies basic Muskingum-Cunge routing against known solutions.
 */

#include <dmc/types.hpp>
#include <dmc/network.hpp>
#include <dmc/router.hpp>
#include <iostream>
#include <cmath>
#include <cassert>
#include <iomanip>

using namespace dmc;

void test_constant_inflow() {
    std::cout << "Testing constant inflow routing... ";
    
    // Single reach
    Network net;
    
    Reach r;
    r.id = 0;
    r.length = 5000.0;  // 5 km
    r.slope = 0.001;
    r.manning_n = Real(0.035);
    r.upstream_junction_id = -1;
    r.downstream_junction_id = 0;
    net.add_reach(r);
    
    Junction j;
    j.id = 0;
    j.is_outlet = true;
    net.add_junction(j);
    
    net.build_topology();
    
    RouterConfig config;
    config.dt = 3600.0;  // 1 hour
    config.enable_gradients = false;
    
    MuskingumCungeRouter router(net, config);
    
    // Constant lateral inflow
    double const_inflow = 10.0;  // m³/s
    router.set_lateral_inflow(0, const_inflow);
    
    // Run until steady state
    double prev_Q = 0.0;
    int max_steps = 1000;
    double tol = 1e-6;
    
    for (int t = 0; t < max_steps; ++t) {
        router.route_timestep();
        double Q = router.get_discharge(0);
        
        if (std::abs(Q - prev_Q) < tol * Q && t > 10) {
            break;
        }
        prev_Q = Q;
    }
    
    // At steady state, outflow should equal inflow (mass conservation)
    double final_Q = router.get_discharge(0);
    assert(std::abs(final_Q - const_inflow) < 0.01 * const_inflow);
    
    std::cout << "PASS (Q_steady = " << final_Q << ", expected ~" << const_inflow << ")\n";
}

void test_pulse_attenuation() {
    std::cout << "Testing pulse attenuation... ";
    
    // Single reach
    Network net;
    
    Reach r;
    r.id = 0;
    r.length = 10000.0;  // 10 km
    r.slope = 0.001;
    r.manning_n = Real(0.035);
    r.upstream_junction_id = -1;
    r.downstream_junction_id = 0;
    net.add_reach(r);
    
    Junction j;
    j.id = 0;
    j.is_outlet = true;
    net.add_junction(j);
    
    net.build_topology();
    
    RouterConfig config;
    config.dt = 900.0;  // 15 min
    config.enable_gradients = false;
    
    MuskingumCungeRouter router(net, config);
    
    // Input pulse
    double peak_input = 50.0;
    int pulse_duration = 4;  // 1 hour
    
    double max_output = 0.0;
    int num_steps = 100;
    
    for (int t = 0; t < num_steps; ++t) {
        double inflow = (t < pulse_duration) ? peak_input : 0.0;
        router.set_lateral_inflow(0, inflow);
        router.route_timestep();
        
        double Q = router.get_discharge(0);
        max_output = std::max(max_output, Q);
    }
    
    // Peak should be attenuated
    assert(max_output < peak_input);
    assert(max_output > 0.0);
    
    std::cout << "PASS (peak_in = " << peak_input << ", peak_out = " << max_output << ")\n";
}

void test_series_routing() {
    std::cout << "Testing series routing (3 reaches)... ";
    
    Network net;
    
    // Three reaches in series
    for (int i = 0; i < 3; ++i) {
        Reach r;
        r.id = i;
        r.length = 5000.0;
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
    
    RouterConfig config;
    config.dt = 3600.0;
    config.enable_gradients = false;
    
    MuskingumCungeRouter router(net, config);
    
    // Constant inflow at first reach only
    router.set_lateral_inflow(0, 10.0);
    router.set_lateral_inflow(1, 0.0);
    router.set_lateral_inflow(2, 0.0);
    
    // Run to steady state
    for (int t = 0; t < 200; ++t) {
        router.route_timestep();
    }
    
    // At steady state, all reaches should have same discharge
    double Q0 = router.get_discharge(0);
    double Q1 = router.get_discharge(1);
    double Q2 = router.get_discharge(2);
    
    assert(std::abs(Q1 - Q0) < 0.01 * Q0);
    assert(std::abs(Q2 - Q0) < 0.01 * Q0);
    
    std::cout << "PASS (Q = " << Q0 << ", " << Q1 << ", " << Q2 << ")\n";
}

void test_muskingum_parameters() {
    std::cout << "Testing Muskingum parameter computation... ";
    
    Network net;
    
    Reach r;
    r.id = 0;
    r.length = 5000.0;
    r.slope = 0.001;
    r.manning_n = Real(0.035);
    r.upstream_junction_id = -1;
    r.downstream_junction_id = 0;
    net.add_reach(r);
    
    Junction j;
    j.id = 0;
    j.is_outlet = true;
    net.add_junction(j);
    
    net.build_topology();
    
    RouterConfig config;
    config.dt = 3600.0;
    config.enable_gradients = false;
    
    MuskingumCungeRouter router(net, config);
    router.set_lateral_inflow(0, 20.0);
    
    // Run a few steps
    for (int t = 0; t < 10; ++t) {
        router.route_timestep();
    }
    
    // Check K and X are reasonable
    const Reach& reach = net.get_reach(0);
    double K = to_double(reach.K);
    double X = to_double(reach.X);
    
    // K should be positive and of order hours for this reach
    assert(K > 0.0);
    assert(K < 86400.0);  // Less than 1 day
    
    // X should be in [0, 0.5]
    assert(X >= 0.0);
    assert(X <= 0.5);
    
    std::cout << "PASS (K = " << K/3600 << " hr, X = " << X << ")\n";
}

int main() {
    std::cout << "=== Single Reach Routing Tests ===\n\n";
    
    std::cout << std::fixed << std::setprecision(3);
    
    try {
        test_constant_inflow();
        test_pulse_attenuation();
        test_series_routing();
        test_muskingum_parameters();
        
        std::cout << "\n=== All single reach tests passed! ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << "\n";
        return 1;
    }
}
