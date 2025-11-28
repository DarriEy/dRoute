/**
 * @file test_comprehensive.cpp
 * @brief Comprehensive test suite for dMC-Route
 * 
 * Tests include:
 * 1. Mass balance verification (all methods)
 * 2. Gradient verification (FD vs AD)
 * 3. Branched network with multiple gauges
 * 4. Method comparison benchmarks
 * 5. Numerical stability tests
 */

 #include <dmc/types.hpp>
 #include <dmc/network.hpp>
 #include <dmc/router.hpp>
 #include <dmc/advanced_routing.hpp>
 #include <iostream>
 #include <iomanip>
 #include <cmath>
 #include <vector>
 #include <chrono>
 #include <cassert>
 
 using namespace dmc;
 
 // ============================================================================
 // Test Network Builders
 // ============================================================================
 
 /**
  * Create a simple linear chain network
  * 
  *   [R1] --> [R2] --> [R3] --> outlet
  */
 Network create_chain_network(int num_reaches = 3) {
     Network net;
     
     for (int i = 0; i < num_reaches; ++i) {
         Reach r;
         r.id = i + 1;
         r.length = 5000.0;  // 5 km
         r.slope = 0.001;
         r.manning_n = Real(0.035);
         r.geometry.width_coef = Real(10.0);
         r.geometry.width_exp = Real(0.4);
         r.geometry.depth_coef = Real(0.5);
         r.geometry.depth_exp = Real(0.3);
         
         if (i > 0) {
             r.upstream_junction_id = i;  // Junction i
         }
         if (i < num_reaches - 1) {
             r.downstream_junction_id = i + 1;  // Junction i+1
         }
         
         net.add_reach(r);
     }
     
     // Create junctions
     for (int i = 1; i < num_reaches; ++i) {
         Junction j;
         j.id = i;
         j.upstream_reach_ids = {i};
         j.downstream_reach_ids = {i + 1};
         net.add_junction(j);
     }
     
     net.build_topology();
     return net;
 }
 
 /**
  * Create a branched (confluence) network
  * 
  *   [R1] ----\
  *             --> [R3] --> outlet
  *   [R2] ----/
  */
 Network create_branched_network() {
     Network net;
     
     // Headwater 1
     Reach r1;
     r1.id = 1;
     r1.length = 8000.0;
     r1.slope = 0.002;
     r1.manning_n = Real(0.030);
     r1.geometry.width_coef = Real(8.0);
     r1.geometry.width_exp = Real(0.45);
     r1.geometry.depth_coef = Real(0.4);
     r1.geometry.depth_exp = Real(0.35);
     r1.downstream_junction_id = 1;
     net.add_reach(r1);
     
     // Headwater 2
     Reach r2;
     r2.id = 2;
     r2.length = 6000.0;
     r2.slope = 0.0015;
     r2.manning_n = Real(0.040);
     r2.geometry.width_coef = Real(6.0);
     r2.geometry.width_exp = Real(0.40);
     r2.geometry.depth_coef = Real(0.35);
     r2.geometry.depth_exp = Real(0.30);
     r2.downstream_junction_id = 1;
     net.add_reach(r2);
     
     // Main channel
     Reach r3;
     r3.id = 3;
     r3.length = 10000.0;
     r3.slope = 0.001;
     r3.manning_n = Real(0.035);
     r3.geometry.width_coef = Real(15.0);
     r3.geometry.width_exp = Real(0.5);
     r3.geometry.depth_coef = Real(0.6);
     r3.geometry.depth_exp = Real(0.3);
     r3.upstream_junction_id = 1;
     net.add_reach(r3);
     
     // Confluence junction
     Junction j1;
     j1.id = 1;
     j1.upstream_reach_ids = {1, 2};
     j1.downstream_reach_ids = {3};
     net.add_junction(j1);
     
     net.build_topology();
     return net;
 }
 
 /**
  * Create a complex network with multiple outlets
  * 
  *   [R1] --> [R3] ----\
  *                      --> [R5] --> outlet1
  *   [R2] --> [R4] ----/
  *                  \
  *                   --> [R6] --> outlet2
  */
 Network create_multi_outlet_network() {
     Network net;
     
     // Build reaches...
     for (int i = 1; i <= 6; ++i) {
         Reach r;
         r.id = i;
         r.length = 5000.0 + i * 1000.0;
         r.slope = 0.001 + (i % 3) * 0.0005;
         r.manning_n = Real(0.030 + (i % 4) * 0.005);
         r.geometry.width_coef = Real(8.0 + i);
         r.geometry.width_exp = Real(0.4);
         r.geometry.depth_coef = Real(0.4);
         r.geometry.depth_exp = Real(0.3);
         net.add_reach(r);
     }
     
     // Set topology
     net.get_reach(1).downstream_junction_id = 1;
     net.get_reach(2).downstream_junction_id = 2;
     net.get_reach(3).upstream_junction_id = 1;
     net.get_reach(3).downstream_junction_id = 3;
     net.get_reach(4).upstream_junction_id = 2;
     net.get_reach(4).downstream_junction_id = 3;
     net.get_reach(5).upstream_junction_id = 3;
     net.get_reach(6).upstream_junction_id = 3;
     
     // Junctions
     Junction j1; j1.id = 1; j1.upstream_reach_ids = {1}; j1.downstream_reach_ids = {3}; net.add_junction(j1);
     Junction j2; j2.id = 2; j2.upstream_reach_ids = {2}; j2.downstream_reach_ids = {4}; net.add_junction(j2);
     Junction j3; j3.id = 3; j3.upstream_reach_ids = {3, 4}; j3.downstream_reach_ids = {5, 6}; net.add_junction(j3);
     
     net.build_topology();
     return net;
 }
 
 // ============================================================================
 // Test 1: Mass Balance Verification
 // ============================================================================
 
 struct MassBalanceResult {
     double total_inflow;
     double total_outflow;
     double final_storage;
     double initial_storage;
     double mass_error;
     double relative_error;
     bool passed;
 };
 
 template<typename Router>
 MassBalanceResult test_mass_balance(Router& router, int num_timesteps, double pulse_magnitude) {
     Network& net = router.network();
     double dt = router.config().dt;
     
     MassBalanceResult result = {};
     
     // Apply pulse inflow at headwaters for first 10 timesteps
     std::vector<int> headwaters;
     for (int id : net.topological_order()) {
         const Reach& r = net.get_reach(id);
         if (r.upstream_junction_id < 0) {
             headwaters.push_back(id);
         }
     }
     
     // Find outlet(s)
     std::vector<int> outlets;
     for (int id : net.topological_order()) {
         const Reach& r = net.get_reach(id);
         if (r.downstream_junction_id < 0) {
             outlets.push_back(id);
         }
     }
     
     // Run simulation - continue until outflow is negligible
     double last_outflow = 0.0;
     int actual_timesteps = 0;
     
     for (int t = 0; t < num_timesteps; ++t) {
         // Apply inflow during pulse period
         for (int hw : headwaters) {
             double inflow = (t < 10) ? pulse_magnitude : 0.0;
             router.set_lateral_inflow(hw, inflow);
             result.total_inflow += inflow * dt;
         }
         
         router.route_timestep();
         actual_timesteps++;
         
         // Accumulate outflow
         last_outflow = 0.0;
         for (int out : outlets) {
             double Q = router.get_discharge(out);
             result.total_outflow += Q * dt;
             last_outflow += Q;
         }
     }
     
     // Final storage estimate: sum of (inflow - outflow) * dt for each reach
     // This represents water still in the reach at the end
     // A better approximation: use final Q values and estimate storage
     result.final_storage = 0.0;
     for (int id : net.topological_order()) {
         const Reach& r = net.get_reach(id);
         double Q = router.get_discharge(id);
         // Estimate storage as Q * travel_time, where travel_time ≈ length / velocity
         // Velocity ≈ Q / (width * depth), but we use a simpler approximation
         // Storage ≈ Q * K, where K is Muskingum time constant
         // K ≈ length / celerity ≈ length / (1.67 * velocity)
         // For simplicity, use Q * (length / 1.0) / 1000 as rough storage estimate
         double storage_time = r.length / 1.0;  // Assume 1 m/s celerity
         result.final_storage += Q * storage_time;
     }
     
     // Mass error: difference between (inflow) and (outflow + remaining storage)
     result.mass_error = std::abs(result.total_outflow + result.final_storage - result.total_inflow);
     result.relative_error = result.mass_error / std::max(result.total_inflow, 1.0);
     result.passed = result.relative_error < 0.05;  // 5% tolerance
     
     return result;
 }
 
 void run_mass_balance_tests() {
     std::cout << "\n========== MASS BALANCE TESTS ==========\n\n";
     
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = false;  // Not needed for mass balance
     
     // Test 1: Chain network with MC
     {
         Network net = create_chain_network(5);
         MuskingumCungeRouter router(net, config);
         auto result = test_mass_balance(router, 100, 50.0);
         
         std::cout << "MC (5-reach chain): ";
         std::cout << "In=" << std::fixed << std::setprecision(1) << result.total_inflow/1e6 << " Mm³, ";
         std::cout << "Out=" << result.total_outflow/1e6 << " Mm³, ";
         std::cout << "Err=" << std::setprecision(2) << result.relative_error * 100 << "% ";
         std::cout << (result.passed ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     // Test 2: Branched network with IRF
     {
         Network net = create_branched_network();
         IRFRouter router(net, config);
         auto result = test_mass_balance(router, 100, 50.0);
         
         std::cout << "IRF (branched):     ";
         std::cout << "In=" << std::fixed << std::setprecision(1) << result.total_inflow/1e6 << " Mm³, ";
         std::cout << "Out=" << result.total_outflow/1e6 << " Mm³, ";
         std::cout << "Err=" << std::setprecision(2) << result.relative_error * 100 << "% ";
         std::cout << (result.passed ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     // Test 3: Chain with KWT
     {
         Network net = create_chain_network(5);
         KWTRouter router(net, config);
         auto result = test_mass_balance(router, 100, 50.0);
         
         std::cout << "KWT (5-reach chain):";
         std::cout << "In=" << std::fixed << std::setprecision(1) << result.total_inflow/1e6 << " Mm³, ";
         std::cout << "Out=" << result.total_outflow/1e6 << " Mm³, ";
         std::cout << "Err=" << std::setprecision(2) << result.relative_error * 100 << "% ";
         std::cout << (result.passed ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     // Test 4: Chain with DW
     {
         Network net = create_chain_network(3);
         DiffusiveWaveRouter router(net, config);
         auto result = test_mass_balance(router, 100, 50.0);
         
         std::cout << "DW (3-reach chain): ";
         std::cout << "In=" << std::fixed << std::setprecision(1) << result.total_inflow/1e6 << " Mm³, ";
         std::cout << "Out=" << result.total_outflow/1e6 << " Mm³, ";
         std::cout << "Err=" << std::setprecision(2) << result.relative_error * 100 << "% ";
         std::cout << (result.passed ? "✓ PASS" : "✗ FAIL") << "\n";
     }
 }
 
 // ============================================================================
 // Test 2: Gradient Verification (Extended)
 // ============================================================================
 
 struct GradientTestResult {
     std::string param_name;
     int reach_id;
     double ad_gradient;
     double fd_gradient;
     double rel_error;
     bool passed;
 };
 
 /**
  * Compute finite difference gradient
  */
 template<typename Router>
 double compute_fd_gradient(Network& net, RouterConfig config, 
                            int reach_id, const std::string& param,
                            double epsilon = 1e-5) {
     // Store original value
     Reach& reach = net.get_reach(reach_id);
     double original;
     
     if (param == "manning_n") {
         original = to_double(reach.manning_n);
     } else if (param == "width_coef") {
         original = to_double(reach.geometry.width_coef);
     } else if (param == "width_exp") {
         original = to_double(reach.geometry.width_exp);
     } else if (param == "depth_coef") {
         original = to_double(reach.geometry.depth_coef);
     } else if (param == "depth_exp") {
         original = to_double(reach.geometry.depth_exp);
     } else {
         return 0.0;
     }
     
     auto run_forward = [&](double value) -> double {
         // Set parameter
         if (param == "manning_n") {
             reach.manning_n = Real(value);
         } else if (param == "width_coef") {
             reach.geometry.width_coef = Real(value);
         } else if (param == "width_exp") {
             reach.geometry.width_exp = Real(value);
         } else if (param == "depth_coef") {
             reach.geometry.depth_coef = Real(value);
         } else if (param == "depth_exp") {
             reach.geometry.depth_exp = Real(value);
         }
         
         // Create fresh router
         config.enable_gradients = false;
         Router router(net, config);
         
         // Apply forcing
         for (int id : net.topological_order()) {
             const Reach& r = net.get_reach(id);
             if (r.upstream_junction_id < 0) {
                 router.set_lateral_inflow(id, 20.0);
             }
         }
         
         // Run
         router.route(30);
         
         // Get outlet discharge
         std::vector<int> outlets;
         for (int id : net.topological_order()) {
             const Reach& r = net.get_reach(id);
             if (r.downstream_junction_id < 0) {
                 outlets.push_back(id);
             }
         }
         
         double total = 0.0;
         for (int out : outlets) {
             total += router.get_discharge(out);
         }
         return total;
     };
     
     // Central finite difference
     double f_plus = run_forward(original + epsilon);
     double f_minus = run_forward(original - epsilon);
     double fd_grad = (f_plus - f_minus) / (2.0 * epsilon);
     
     // Restore original
     if (param == "manning_n") {
         reach.manning_n = Real(original);
     } else if (param == "width_coef") {
         reach.geometry.width_coef = Real(original);
     } else if (param == "width_exp") {
         reach.geometry.width_exp = Real(original);
     } else if (param == "depth_coef") {
         reach.geometry.depth_coef = Real(original);
     } else if (param == "depth_exp") {
         reach.geometry.depth_exp = Real(original);
     }
     
     return fd_grad;
 }
 
 void run_gradient_tests() {
     std::cout << "\n========== GRADIENT VERIFICATION TESTS ==========\n\n";
     
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = true;
     
     std::vector<std::string> params = {"manning_n", "width_coef", "depth_coef"};
     double tolerance = 0.20;  // 20% relative error tolerance
     
     // Test MC on branched network
     std::cout << "Muskingum-Cunge (branched network):\n";
     std::cout << "Reach  Parameter      AD Grad        FD Grad        Rel Err    Status\n";
     std::cout << "----------------------------------------------------------------------\n";
     
     {
         Network net = create_branched_network();
         
         for (int reach_id : {1, 2, 3}) {
             for (const auto& param : params) {
                 // Compute FD gradient
                 double fd_grad = compute_fd_gradient<MuskingumCungeRouter>(net, config, reach_id, param);
                 
                 // Compute AD gradient
                 config.enable_gradients = true;
                 MuskingumCungeRouter router(net, config);
                 
                 for (int id : net.topological_order()) {
                     const Reach& r = net.get_reach(id);
                     if (r.upstream_junction_id < 0) {
                         router.set_lateral_inflow(id, 20.0);
                     }
                 }
                 
                 router.start_recording();
                 router.route(30);
                 router.stop_recording();
                 
                 std::vector<int> outlets = {3};
                 std::vector<double> dL_dQ = {1.0};
                 router.compute_gradients(outlets, dL_dQ);
                 
                 auto grads = router.get_gradients();
                 std::string key = "reach_" + std::to_string(reach_id) + "_" + param;
                 double ad_grad = grads.count(key) ? grads[key] : 0.0;
                 
                 // Compute relative error
                 double rel_err = 0.0;
                 if (std::abs(fd_grad) > 1e-10 || std::abs(ad_grad) > 1e-10) {
                     rel_err = std::abs(ad_grad - fd_grad) / std::max(std::abs(fd_grad), std::abs(ad_grad));
                 }
                 
                 bool passed = rel_err < tolerance || (std::abs(fd_grad) < 1e-10 && std::abs(ad_grad) < 1e-10);
                 
                 std::cout << std::setw(5) << reach_id << "  "
                           << std::setw(12) << param << "  "
                           << std::scientific << std::setprecision(2)
                           << std::setw(12) << ad_grad << "  "
                           << std::setw(12) << fd_grad << "  "
                           << std::fixed << std::setprecision(1)
                           << std::setw(8) << rel_err * 100 << "%  "
                           << (passed ? "✓ PASS" : "✗ FAIL") << "\n";
             }
         }
     }
     
     // Test IRF gradients
     std::cout << "\nIRF (chain network):\n";
     std::cout << "Reach  Parameter      AD Grad        FD Grad        Rel Err    Status\n";
     std::cout << "----------------------------------------------------------------------\n";
     
     {
         Network net = create_chain_network(3);
         
         for (int reach_id : {1, 2, 3}) {
             double fd_grad = compute_fd_gradient<IRFRouter>(net, config, reach_id, "manning_n");
             
             config.enable_gradients = true;
             IRFRouter router(net, config);
             router.set_lateral_inflow(1, 20.0);
             
             router.start_recording();
             router.route(30);
             router.stop_recording();
             
             std::vector<int> outlets = {3};
             std::vector<double> dL_dQ = {1.0};
             router.compute_gradients(outlets, dL_dQ);
             
             auto grads = router.get_gradients();
             std::string key = "reach_" + std::to_string(reach_id) + "_manning_n";
             double ad_grad = grads.count(key) ? grads[key] : 0.0;
             
             double rel_err = 0.0;
             if (std::abs(fd_grad) > 1e-10 || std::abs(ad_grad) > 1e-10) {
                 rel_err = std::abs(ad_grad - fd_grad) / std::max(std::abs(fd_grad), std::abs(ad_grad));
             }
             
             bool passed = rel_err < tolerance || (std::abs(fd_grad) < 1e-10 && std::abs(ad_grad) < 1e-10);
             
             std::cout << std::setw(5) << reach_id << "  "
                       << std::setw(12) << "manning_n" << "  "
                       << std::scientific << std::setprecision(2)
                       << std::setw(12) << ad_grad << "  "
                       << std::setw(12) << fd_grad << "  "
                       << std::fixed << std::setprecision(1)
                       << std::setw(8) << rel_err * 100 << "%  "
                       << (passed ? "✓ PASS" : "✗ FAIL") << "\n";
         }
     }
 }
 
 // ============================================================================
 // Test 3: Method Comparison Benchmark
 // ============================================================================
 
 void run_method_comparison() {
     std::cout << "\n========== METHOD COMPARISON ==========\n\n";
     
     Network net = create_chain_network(10);
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = false;
     
     int num_timesteps = 200;
     
     std::cout << "Network: 10-reach chain, 200 timesteps\n";
     std::cout << "Method              Peak Q (m³/s)    Time to Peak (h)    Runtime (ms)\n";
     std::cout << "-----------------------------------------------------------------------\n";
     
     auto run_and_measure = [&](const std::string& name, auto& router) {
         // Apply pulse forcing
         router.set_lateral_inflow(1, 100.0);
         
         auto start = std::chrono::high_resolution_clock::now();
         
         double peak_Q = 0.0;
         int peak_t = 0;
         
         for (int t = 0; t < num_timesteps; ++t) {
             if (t == 20) router.set_lateral_inflow(1, 0.0);  // End pulse
             router.route_timestep();
             
             double Q = router.get_discharge(10);  // Outlet
             if (Q > peak_Q) {
                 peak_Q = Q;
                 peak_t = t;
             }
         }
         
         auto end = std::chrono::high_resolution_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
         
         std::cout << std::setw(18) << name << "  "
                   << std::fixed << std::setprecision(1) << std::setw(14) << peak_Q << "  "
                   << std::setw(16) << peak_t << "  "
                   << std::setw(12) << duration.count() / 1000.0 << "\n";
     };
     
     {
         Network net_copy = create_chain_network(10);
         MuskingumCungeRouter router(net_copy, config);
         run_and_measure("Muskingum-Cunge", router);
     }
     
     {
         Network net_copy = create_chain_network(10);
         IRFRouter router(net_copy, config);
         run_and_measure("IRF", router);
     }
     
     {
         Network net_copy = create_chain_network(10);
         DiffusiveWaveRouter router(net_copy, config);
         run_and_measure("Diffusive Wave", router);
     }
     
     {
         Network net_copy = create_chain_network(10);
         LagRouter router(net_copy, config);
         run_and_measure("Lag", router);
     }
     
     {
         Network net_copy = create_chain_network(10);
         KWTRouter router(net_copy, config);
         run_and_measure("KWT", router);
     }
 }
 
 // ============================================================================
 // Test 4: Numerical Stability
 // ============================================================================
 
 void run_stability_tests() {
     std::cout << "\n========== NUMERICAL STABILITY TESTS ==========\n\n";
     
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = true;
     
     int passed = 0;
     int total = 0;
     
     // Test 1: Very low flow
     {
         Network net = create_chain_network(3);
         MuskingumCungeRouter router(net, config);
         
         router.set_lateral_inflow(1, 0.001);  // Very low flow
         
         bool stable = true;
         for (int t = 0; t < 50; ++t) {
             router.route_timestep();
             double Q = router.get_discharge(3);
             if (std::isnan(Q) || std::isinf(Q) || Q < 0) {
                 stable = false;
                 break;
             }
         }
         
         total++;
         if (stable) passed++;
         std::cout << "Low flow (0.001 m³/s):  " << (stable ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     // Test 2: Very high flow
     {
         Network net = create_chain_network(3);
         MuskingumCungeRouter router(net, config);
         
         router.set_lateral_inflow(1, 10000.0);  // Very high flow
         
         bool stable = true;
         for (int t = 0; t < 50; ++t) {
             router.route_timestep();
             double Q = router.get_discharge(3);
             if (std::isnan(Q) || std::isinf(Q) || Q < 0) {
                 stable = false;
                 break;
             }
         }
         
         total++;
         if (stable) passed++;
         std::cout << "High flow (10000 m³/s): " << (stable ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     // Test 3: Zero lateral inflow
     {
         Network net = create_chain_network(3);
         MuskingumCungeRouter router(net, config);
         
         // No lateral inflow at all
         
         bool stable = true;
         for (int t = 0; t < 50; ++t) {
             router.route_timestep();
             double Q = router.get_discharge(3);
             if (std::isnan(Q) || std::isinf(Q) || Q < 0) {
                 stable = false;
                 break;
             }
         }
         
         total++;
         if (stable) passed++;
         std::cout << "Zero inflow:            " << (stable ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     // Test 4: Steep slope
     {
         Network net;
         Reach r;
         r.id = 1;
         r.length = 1000.0;
         r.slope = 0.1;  // 10% slope!
         r.manning_n = Real(0.035);
         net.add_reach(r);
         net.build_topology();
         
         MuskingumCungeRouter router(net, config);
         router.set_lateral_inflow(1, 50.0);
         
         bool stable = true;
         for (int t = 0; t < 50; ++t) {
             router.route_timestep();
             double Q = router.get_discharge(1);
             if (std::isnan(Q) || std::isinf(Q) || Q < 0) {
                 stable = false;
                 break;
             }
         }
         
         total++;
         if (stable) passed++;
         std::cout << "Steep slope (10%):      " << (stable ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     // Test 5: Very flat slope
     {
         Network net;
         Reach r;
         r.id = 1;
         r.length = 10000.0;
         r.slope = 0.00001;  // Very flat
         r.manning_n = Real(0.035);
         net.add_reach(r);
         net.build_topology();
         
         MuskingumCungeRouter router(net, config);
         router.set_lateral_inflow(1, 50.0);
         
         bool stable = true;
         for (int t = 0; t < 50; ++t) {
             router.route_timestep();
             double Q = router.get_discharge(1);
             if (std::isnan(Q) || std::isinf(Q) || Q < 0) {
                 stable = false;
                 break;
             }
         }
         
         total++;
         if (stable) passed++;
         std::cout << "Flat slope (0.001%):    " << (stable ? "✓ PASS" : "✗ FAIL") << "\n";
     }
     
     std::cout << "\nStability: " << passed << "/" << total << " tests passed\n";
 }
 
 // ============================================================================
 // Test 5: Soft-KWT Steepness Sensitivity
 // ============================================================================
 
 void run_steepness_sensitivity_test() {
     std::cout << "\n========== KWT-SOFT STEEPNESS SENSITIVITY ==========\n\n";
     
     Network net = create_chain_network(3);
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = true;
     
     std::cout << "Testing gradient magnitude vs steepness...\n";
     std::cout << "Steepness    Peak Q (m³/s)    Grad magnitude    Grad sign\n";
     std::cout << "------------------------------------------------------------\n";
     
     for (double steepness : {1.0, 2.0, 5.0, 10.0, 20.0}) {
         config.kwt_gate_steepness = steepness;
         
         Network net_copy = create_chain_network(3);
         SoftGatedKWT router(net_copy, config);
         router.set_steepness(steepness);
         
         router.set_lateral_inflow(1, 50.0);
         
         router.start_recording();
         double peak_Q = 0.0;
         for (int t = 0; t < 50; ++t) {
             if (t == 10) router.set_lateral_inflow(1, 0.0);
             router.route_timestep();
             peak_Q = std::max(peak_Q, router.get_discharge(3));
         }
         router.stop_recording();
         
         std::vector<int> outlets = {3};
         std::vector<double> dL_dQ = {1.0};
         router.compute_gradients(outlets, dL_dQ);
         
         auto grads = router.get_gradients();
         double grad_mag = 0.0;
         for (const auto& [key, val] : grads) {
             grad_mag += val * val;
         }
         grad_mag = std::sqrt(grad_mag);
         
         // Check gradient sign (should be negative for manning_n)
         double grad_n = grads.count("reach_1_manning_n") ? grads["reach_1_manning_n"] : 0.0;
         std::string sign = grad_n < 0 ? "negative ✓" : (grad_n == 0 ? "zero" : "positive ✗");
         
         std::cout << std::fixed << std::setprecision(1)
                   << std::setw(9) << steepness << "  "
                   << std::setw(14) << peak_Q << "  "
                   << std::scientific << std::setprecision(2)
                   << std::setw(14) << grad_mag << "  "
                   << sign << "\n";
     }
     
     std::cout << "\nNote: Very high steepness may cause vanishing gradients.\n";
     std::cout << "For training, start with low steepness and anneal upward.\n";
 }
 
 // ============================================================================
 // Main
 // ============================================================================
 
 int main() {
     std::cout << R"(
 ╔═════════════════════════════════════════════════════════════════════╗
 ║  dMC-Route Comprehensive Test Suite                                  ║
 ║  Tests: Mass Balance, Gradients, Stability, Methods, Steepness      ║
 ╚═════════════════════════════════════════════════════════════════════╝
 )" << std::endl;
 
     run_mass_balance_tests();
     run_gradient_tests();
     run_method_comparison();
     run_stability_tests();
     run_steepness_sensitivity_test();
     
     std::cout << "\n========== ALL TESTS COMPLETE ==========\n";
     
     return 0;
 }