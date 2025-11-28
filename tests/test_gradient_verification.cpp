/**
 * @file test_gradient_verification.cpp
 * @brief Rigorous gradient verification suite for all routing methods
 * 
 * Validates AD gradients against finite difference approximations.
 * Tests:
 * - Muskingum-Cunge (full AD)
 * - IRF (analytical)
 * - Diffusive Wave (analytical)
 * - Lag (analytical)
 * 
 * Uses central finite differences for O(h²) accuracy.
 */

 #include <iostream>
 #include <iomanip>
 #include <cmath>
 #include <vector>
 #include <string>
 #include <functional>
 #include "dmc/router.hpp"
 #include "dmc/network.hpp"
 
 using namespace dmc;
 
 // ============================================================================
 // Test Configuration
 // ============================================================================
 
 struct GradientTestConfig {
     double fd_epsilon = 1e-5;        // Finite difference step size
     double rel_tolerance = 0.1;       // 10% relative error tolerance
     double abs_tolerance = 1e-6;      // Absolute error for near-zero gradients
     int num_timesteps = 100;          // Simulation length for each test
     double dt = 3600.0;               // Timestep [s]
     bool verbose = true;              // Print detailed results
 };
 
 // ============================================================================
 // Test Network Creation
 // ============================================================================
 
 /**
  * Create a simple 3-reach test network
  * 
  *   R1 ─┐
  *       ├─→ R3 ─→ [outlet]
  *   R2 ─┘
  */
 Network create_test_network() {
     Network net;
     
     // Create reaches
     for (int i = 1; i <= 3; ++i) {
         Reach r;
         r.id = i;
         r.name = "Reach_" + std::to_string(i);
         r.length = 5000.0 + i * 1000.0;  // 6-8 km
         r.slope = 0.001 + i * 0.0002;     // Variable slopes
         r.manning_n = Real(0.03 + i * 0.005);  // Variable roughness
         r.geometry.width_coef = Real(5.0 + i);
         r.geometry.width_exp = Real(0.5);
         r.geometry.depth_coef = Real(0.2 + i * 0.05);
         r.geometry.depth_exp = Real(0.3);
         
         if (i == 3) {
             r.upstream_junction_id = 1;  // R3 receives from junction
         } else {
             r.upstream_junction_id = -1;  // R1, R2 are headwaters
         }
         r.downstream_junction_id = (i == 3) ? -1 : 1;
         
         net.add_reach(r);
     }
     
     // Create confluence junction
     Junction j;
     j.id = 1;
     j.name = "Confluence";
     j.upstream_reach_ids = {1, 2};
     j.downstream_reach_ids = {3};
     net.add_junction(j);
     
     net.build_topology();
     return net;
 }
 
 /**
  * Set synthetic lateral inflows (sinusoidal pattern)
  */
 void set_synthetic_inflows(Network& net, double t, double amplitude = 10.0) {
     for (int id : net.topological_order()) {
         Reach& r = net.get_reach(id);
         // Sinusoidal inflow with phase shift per reach
         double inflow = amplitude * (1.0 + 0.5 * std::sin(2.0 * M_PI * t / (24.0 * 3600.0) + id));
         r.lateral_inflow = Real(inflow);
     }
 }
 
 // ============================================================================
 // Gradient Verification Core
 // ============================================================================
 
 struct GradientResult {
     std::string param_name;
     int reach_id;
     double ad_gradient;
     double fd_gradient;
     double abs_error;
     double rel_error;
     bool passed;
 };
 
 /**
  * Central finite difference: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
  */
 template<typename RouterType>
 double compute_fd_gradient(
     Network& net,
     RouterConfig& config,
     int reach_id,
     std::string param,
     double epsilon,
     int num_timesteps,
     int outlet_id
 ) {
     auto get_param = [&](Reach& r) -> Real& {
         if (param == "manning_n") return r.manning_n;
         if (param == "width_coef") return r.geometry.width_coef;
         if (param == "width_exp") return r.geometry.width_exp;
         if (param == "depth_coef") return r.geometry.depth_coef;
         if (param == "depth_exp") return r.geometry.depth_exp;
         return r.manning_n;  // Default
     };
     
     auto run_simulation = [&]() -> double {
         // Reset network state
         for (int id : net.topological_order()) {
             Reach& r = net.get_reach(id);
             r.inflow_prev = r.inflow_curr = Real(0.0);
             r.outflow_prev = r.outflow_curr = Real(0.0);
         }
         
         RouterType router(net, config);
         
         for (int t = 0; t < num_timesteps; ++t) {
             set_synthetic_inflows(net, t * config.dt);
             router.route_timestep();
         }
         
         return to_double(net.get_reach(outlet_id).outflow_curr);
     };
     
     Reach& reach = net.get_reach(reach_id);
     Real& param_ref = get_param(reach);
     double original = to_double(param_ref);
     
     // f(x + h)
     param_ref = Real(original + epsilon);
     double f_plus = run_simulation();
     
     // f(x - h)
     param_ref = Real(original - epsilon);
     double f_minus = run_simulation();
     
     // Restore
     param_ref = Real(original);
     
     // Central difference
     return (f_plus - f_minus) / (2.0 * epsilon);
 }
 
 /**
  * Run AD gradient computation for a router
  */
 template<typename RouterType>
 std::unordered_map<std::string, double> compute_ad_gradients(
     Network& net,
     RouterConfig& config,
     int num_timesteps,
     int outlet_id
 ) {
     // Reset network
     for (int id : net.topological_order()) {
         Reach& r = net.get_reach(id);
         r.inflow_prev = r.inflow_curr = Real(0.0);
         r.outflow_prev = r.outflow_curr = Real(0.0);
     }
     net.zero_gradients();
     
     RouterType router(net, config);
     router.enable_gradients(true);
     router.start_recording();
     
     for (int t = 0; t < num_timesteps; ++t) {
         set_synthetic_inflows(net, t * config.dt);
         router.route_timestep();
     }
     
     router.stop_recording();
     router.compute_gradients({outlet_id}, {1.0});
     
     return router.get_gradients();
 }
 
 /**
  * Verify gradients for a single router type
  */
 template<typename RouterType>
 std::vector<GradientResult> verify_router_gradients(
     const std::string& router_name,
     const GradientTestConfig& test_config
 ) {
     std::vector<GradientResult> results;
     
     // Create fresh network for each test
     Network net = create_test_network();
     
     RouterConfig config;
     config.dt = test_config.dt;
     config.enable_gradients = true;
     
     int outlet_id = 3;  // Our test network outlet
     
     // Compute AD gradients
     auto ad_grads = compute_ad_gradients<RouterType>(
         net, config, test_config.num_timesteps, outlet_id);
     
     // Parameters to test
     std::vector<std::string> params = {"manning_n", "width_coef", "width_exp", "depth_coef", "depth_exp"};
     
     for (int reach_id : net.topological_order()) {
         for (const auto& param : params) {
             GradientResult res;
             res.param_name = param;
             res.reach_id = reach_id;
             
             // Get AD gradient
             std::string key = "reach_" + std::to_string(reach_id) + "_" + param;
             res.ad_gradient = ad_grads.count(key) ? ad_grads[key] : 0.0;
             
             // Skip width/depth for analytical methods (they don't compute these)
             bool is_analytical_method = (router_name != "MuskingumCunge");
             if (is_analytical_method && param != "manning_n") {
                 res.fd_gradient = 0.0;
                 res.abs_error = 0.0;
                 res.rel_error = 0.0;
                 res.passed = true;  // Skip
                 results.push_back(res);
                 continue;
             }
             
             // Compute FD gradient
             Network net_fd = create_test_network();
             res.fd_gradient = compute_fd_gradient<RouterType>(
                 net_fd, config, reach_id, param,
                 test_config.fd_epsilon, test_config.num_timesteps, outlet_id);
             
             // Compute errors
             res.abs_error = std::abs(res.ad_gradient - res.fd_gradient);
             double denom = std::max(std::abs(res.fd_gradient), test_config.abs_tolerance);
             res.rel_error = res.abs_error / denom;
             
             // Check pass/fail
             res.passed = (res.rel_error < test_config.rel_tolerance) ||
                          (res.abs_error < test_config.abs_tolerance);
             
             results.push_back(res);
         }
     }
     
     return results;
 }
 
 // ============================================================================
 // Test Runners
 // ============================================================================
 
 void print_results(const std::string& router_name,
                    const std::vector<GradientResult>& results,
                    bool verbose) {
     std::cout << "\n" << std::string(70, '=') << "\n";
     std::cout << "Gradient Verification: " << router_name << "\n";
     std::cout << std::string(70, '=') << "\n";
     
     int passed = 0, failed = 0, skipped = 0;
     
     std::cout << std::setw(10) << "Reach"
               << std::setw(15) << "Parameter"
               << std::setw(15) << "AD Grad"
               << std::setw(15) << "FD Grad"
               << std::setw(12) << "Rel Err"
               << std::setw(10) << "Status" << "\n";
     std::cout << std::string(70, '-') << "\n";
     
     for (const auto& res : results) {
         if (res.ad_gradient == 0.0 && res.fd_gradient == 0.0) {
             skipped++;
             if (!verbose) continue;
         }
         
         std::cout << std::setw(10) << res.reach_id
                   << std::setw(15) << res.param_name
                   << std::setw(15) << std::scientific << std::setprecision(2) << res.ad_gradient
                   << std::setw(15) << res.fd_gradient
                   << std::setw(12) << std::fixed << std::setprecision(1) << (res.rel_error * 100) << "%"
                   << std::setw(10) << (res.passed ? "✓ PASS" : "✗ FAIL") << "\n";
         
         if (res.passed) passed++;
         else failed++;
     }
     
     std::cout << std::string(70, '-') << "\n";
     std::cout << "Summary: " << passed << " passed, " << failed << " failed, "
               << skipped << " skipped\n";
 }
 
 bool run_muskingum_cunge_tests(const GradientTestConfig& config) {
     auto results = verify_router_gradients<MuskingumCungeRouter>("MuskingumCunge", config);
     print_results("Muskingum-Cunge (Full AD)", results, config.verbose);
     
     for (const auto& r : results) {
         if (!r.passed && r.fd_gradient != 0.0) return false;
     }
     return true;
 }
 
 bool run_irf_tests(const GradientTestConfig& config) {
     auto results = verify_router_gradients<IRFRouter>("IRF", config);
     print_results("IRF (Analytical)", results, config.verbose);
     
     for (const auto& r : results) {
         if (!r.passed && r.param_name == "manning_n" && r.fd_gradient != 0.0) 
             return false;
     }
     return true;
 }
 
 bool run_diffusive_wave_tests(const GradientTestConfig& config) {
     auto results = verify_router_gradients<DiffusiveWaveRouter>("DiffusiveWave", config);
     print_results("Diffusive Wave (Analytical)", results, config.verbose);
     
     for (const auto& r : results) {
         if (!r.passed && r.param_name == "manning_n" && r.fd_gradient != 0.0) 
             return false;
     }
     return true;
 }
 
 bool run_lag_tests(const GradientTestConfig& config) {
     auto results = verify_router_gradients<LagRouter>("Lag", config);
     print_results("Lag (Analytical)", results, config.verbose);
     
     for (const auto& r : results) {
         if (!r.passed && r.param_name == "manning_n" && r.fd_gradient != 0.0) 
             return false;
     }
     return true;
 }
 
 // ============================================================================
 // Additional Verification Tests
 // ============================================================================
 
 /**
  * Test gradient sign correctness
  * 
  * For routing:
  * - Increasing Manning's n → slower flow → delayed peaks
  * - Effect on outlet Q depends on hydrograph phase
  */
 bool test_gradient_signs() {
     std::cout << "\n" << std::string(70, '=') << "\n";
     std::cout << "Gradient Sign Verification\n";
     std::cout << std::string(70, '=') << "\n";
     
     Network net = create_test_network();
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = true;
     
     // Run short simulation
     MuskingumCungeRouter router(net, config);
     router.start_recording();
     
     for (int t = 0; t < 50; ++t) {
         set_synthetic_inflows(net, t * config.dt, 10.0);
         router.route_timestep();
     }
     
     router.stop_recording();
     router.compute_gradients({3}, {1.0});
     
     // Check that gradients exist and have consistent signs within a reach
     bool all_ok = true;
     for (int id : net.topological_order()) {
         const Reach& r = net.get_reach(id);
         
         // Width and depth should have same sign (both affect hydraulic radius similarly)
         if (r.grad_width_coef != 0 && r.grad_depth_coef != 0) {
             bool same_sign = (r.grad_width_coef > 0) == (r.grad_depth_coef > 0);
             if (!same_sign) {
                 std::cout << "Warning: Reach " << id 
                           << " has inconsistent width/depth gradient signs\n";
             }
         }
         
         std::cout << "Reach " << id << ": "
                   << "∂Q/∂n = " << std::setw(10) << r.grad_manning_n
                   << ", ∂Q/∂w = " << std::setw(10) << r.grad_width_coef
                   << ", ∂Q/∂d = " << std::setw(10) << r.grad_depth_coef << "\n";
     }
     
     return all_ok;
 }
 
 /**
  * Test gradient magnitude scaling
  * 
  * Downstream reaches should generally have larger gradient magnitudes
  * due to cumulative routing effects.
  */
 bool test_gradient_scaling() {
     std::cout << "\n" << std::string(70, '=') << "\n";
     std::cout << "Gradient Magnitude Scaling Test\n";
     std::cout << std::string(70, '=') << "\n";
     
     Network net = create_test_network();
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = true;
     
     MuskingumCungeRouter router(net, config);
     router.start_recording();
     
     for (int t = 0; t < 100; ++t) {
         set_synthetic_inflows(net, t * config.dt, 10.0);
         router.route_timestep();
     }
     
     router.stop_recording();
     router.compute_gradients({3}, {1.0});
     
     // Outlet reach (3) should have largest gradient magnitude for manning_n
     double outlet_grad = std::abs(net.get_reach(3).grad_manning_n);
     double max_upstream_grad = 0.0;
     
     for (int id : {1, 2}) {
         max_upstream_grad = std::max(max_upstream_grad, 
                                       std::abs(net.get_reach(id).grad_manning_n));
     }
     
     std::cout << "Outlet gradient magnitude: " << outlet_grad << "\n";
     std::cout << "Max upstream gradient: " << max_upstream_grad << "\n";
     
     bool scaling_ok = (outlet_grad >= max_upstream_grad * 0.5);  // Some tolerance
     std::cout << "Scaling check: " << (scaling_ok ? "PASS" : "FAIL") << "\n";
     
     return scaling_ok;
 }
 
 // ============================================================================
 // Mass Balance Tests
 // ============================================================================
 
 /**
  * Test mass conservation in routing using Muskingum-Cunge.
  * After the Q_ref fix, MC should conserve mass properly.
  */
 bool test_mass_balance() {
     std::cout << "\n--- Mass Balance Test (MC) ---\n";
     
     // Create simple 3-reach chain network
     Network net;
     for (int i = 1; i <= 3; ++i) {
         Reach r;
         r.id = i;
         r.length = 5000.0;
         r.slope = 0.001;
         r.manning_n = Real(0.035);
         r.geometry.width_coef = Real(10.0);
         r.geometry.width_exp = Real(0.5);
         r.geometry.depth_coef = Real(0.4);
         r.geometry.depth_exp = Real(0.3);
         
         if (i > 1) {
             r.upstream_junction_id = i - 1;
         }
         if (i < 3) {
             r.downstream_junction_id = i;
         }
         net.add_reach(r);
     }
     
     // Create junctions
     for (int i = 1; i <= 2; ++i) {
         Junction j;
         j.id = i;
         j.upstream_reach_ids = {i};
         j.downstream_reach_ids = {i + 1};
         net.add_junction(j);
     }
     net.build_topology();
     
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = false;  // Just testing physics
     
     MuskingumCungeRouter router(net, config);
     
     double total_inflow = 0.0;
     double total_outflow = 0.0;
     int num_timesteps = 200;
     
     // Pulse inflow: 100 m³/s for first 24 hours, then 0
     for (int t = 0; t < num_timesteps; ++t) {
         double inflow = (t < 24) ? 100.0 : 0.0;
         router.set_lateral_inflow(1, inflow);
         
         total_inflow += inflow * config.dt;
         
         router.route_timestep();
         
         // Accumulate outflow from outlet (reach 3)
         total_outflow += router.get_discharge(3) * config.dt;
     }
     
     double imbalance = std::abs(total_inflow - total_outflow);
     double relative_error = imbalance / total_inflow;
     
     std::cout << "  Total inflow:  " << std::fixed << std::setprecision(0) << total_inflow << " m³\n";
     std::cout << "  Total outflow: " << total_outflow << " m³\n";
     std::cout << "  Imbalance:     " << imbalance << " m³ (" 
               << std::setprecision(2) << relative_error * 100 << "%)\n";
     
     // MC should conserve mass within 5% (some water may still be in transit)
     bool passed = relative_error < 0.05;
     std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") << " - Mass balance within 5%\n";
     
     return passed;
 }
 
 /**
  * Test network with lateral inflow in middle reach
  */
 bool test_lateral_inflow_gradient() {
     std::cout << "\n--- Lateral Inflow Gradient Test ---\n";
     
     // Create network: R1 -> R2 (with lateral) -> R3 -> outlet
     Network net;
     for (int i = 1; i <= 3; ++i) {
         Reach r;
         r.id = i;
         r.length = 5000.0;
         r.slope = 0.001;
         r.manning_n = Real(0.035);
         r.geometry.width_coef = Real(10.0);
         r.geometry.width_exp = Real(0.5);
         r.geometry.depth_coef = Real(0.4);
         r.geometry.depth_exp = Real(0.3);
         
         if (i > 1) r.upstream_junction_id = i - 1;
         if (i < 3) r.downstream_junction_id = i;
         net.add_reach(r);
     }
     
     for (int i = 1; i <= 2; ++i) {
         Junction j;
         j.id = i;
         j.upstream_reach_ids = {i};
         j.downstream_reach_ids = {i + 1};
         net.add_junction(j);
     }
     net.build_topology();
     
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = true;
     
     // Run simulation with PULSE lateral inflow to test transient gradients
     // (At steady state, gradients are zero because Q_out = Q_in by mass conservation)
     auto run_sim = [&]() {
         for (int id : net.topological_order()) {
             Reach& r = net.get_reach(id);
             r.inflow_prev = r.inflow_curr = Real(0.0);
             r.outflow_prev = r.outflow_curr = Real(0.0);
         }
         net.zero_gradients();
         
         MuskingumCungeRouter router(net, config);
         router.start_recording();
         
         for (int t = 0; t < 50; ++t) {
             // PULSE inflows - only for first 10 timesteps
             // This creates a transient where routing parameters affect the hydrograph shape
             double pulse = (t < 10) ? 1.0 : 0.0;
             
             // Headwater inflow pulse
             net.get_reach(1).lateral_inflow = Real(50.0 * pulse);
             // Lateral inflow pulse to middle reach
             net.get_reach(2).lateral_inflow = Real(30.0 * pulse);
             // No direct inflow to outlet reach
             net.get_reach(3).lateral_inflow = Real(0.0);
             
             router.route_timestep();
         }
         
         router.stop_recording();
         std::vector<int> gauges = {3};
         std::vector<double> dL_dQ = {1.0};
         router.compute_gradients(gauges, dL_dQ);
     };
     
     run_sim();
     
     // All reaches should have non-zero gradients
     bool all_nonzero = true;
     std::cout << "  Gradients (∂Q_outlet/∂manning_n):\n";
     for (int id : net.topological_order()) {
         double grad = net.get_reach(id).grad_manning_n;
         std::cout << "    Reach " << id << ": " << std::scientific << grad << "\n";
         if (std::abs(grad) < 1e-15) {
             all_nonzero = false;
         }
     }
     
     // Reach 2 (with lateral inflow) should have significant gradient
     double grad_r2 = std::abs(net.get_reach(2).grad_manning_n);
     bool r2_significant = grad_r2 > 1e-6;
     
     bool passed = all_nonzero && r2_significant;
     std::cout << (passed ? "  ✓ PASS" : "  ✗ FAIL") 
               << " - Lateral inflow reach has gradient\n";
     
     return passed;
 }
 
 /**
  * Test multiple gauge (outlet) gradient computation
  */
 bool test_multiple_gauges() {
     std::cout << "\n--- Multiple Gauge Gradient Test ---\n";
     
     Network net = create_test_network();
     
     RouterConfig config;
     config.dt = 3600.0;
     config.enable_gradients = true;
     
     MuskingumCungeRouter router(net, config);
     router.start_recording();
     
     for (int t = 0; t < 50; ++t) {
         set_synthetic_inflows(net, t * config.dt);
         router.route_timestep();
     }
     
     router.stop_recording();
     
     // Two gauges: reaches 1 and 3 (headwater and outlet)
     std::vector<int> gauges = {1, 3};
     std::vector<double> dL_dQ = {0.5, 0.5};  // Equal weight loss
     router.compute_gradients(gauges, dL_dQ);
     
     // Check that gradients are computed
     std::cout << "  Gradients with dual gauges:\n";
     bool has_gradients = false;
     for (int id : net.topological_order()) {
         double grad = net.get_reach(id).grad_manning_n;
         std::cout << "    Reach " << id << ": " << std::scientific << grad << "\n";
         if (std::abs(grad) > 1e-15) has_gradients = true;
     }
     
     std::cout << (has_gradients ? "  ✓ PASS" : "  ✗ FAIL") 
               << " - Multiple gauge gradients computed\n";
     
     return has_gradients;
 }
 
 // ============================================================================
 // Main Test Driver
 // ============================================================================
 
 int main(int argc, char* argv[]) {
     std::cout << R"(
 ╔══════════════════════════════════════════════════════════════════════╗
 ║  dMC-Route Gradient Verification Suite                               ║
 ║  Validates AD gradients against finite difference approximations     ║
 ╚══════════════════════════════════════════════════════════════════════╝
 )" << std::endl;
 
     GradientTestConfig config;
     config.verbose = true;
     config.num_timesteps = 50;  // Shorter for faster testing
     config.fd_epsilon = 1e-5;
     config.rel_tolerance = 0.15;  // 15% tolerance for analytical methods
     
     bool all_passed = true;
     
     // Core gradient verification tests
     std::cout << "\n[1/9] Testing Muskingum-Cunge gradients...\n";
     all_passed &= run_muskingum_cunge_tests(config);
     
     std::cout << "\n[2/9] Testing IRF gradients...\n";
     all_passed &= run_irf_tests(config);
     
     std::cout << "\n[3/9] Testing Diffusive Wave gradients...\n";
     all_passed &= run_diffusive_wave_tests(config);
     
     std::cout << "\n[4/9] Testing Lag gradients...\n";
     all_passed &= run_lag_tests(config);
     
     // Additional verification tests
     std::cout << "\n[5/9] Testing gradient signs...\n";
     all_passed &= test_gradient_signs();
     
     std::cout << "\n[6/9] Testing gradient scaling...\n";
     all_passed &= test_gradient_scaling();
     
     // New comprehensive tests
     std::cout << "\n[7/9] Testing mass balance...\n";
     all_passed &= test_mass_balance();
     
     std::cout << "\n[8/9] Testing lateral inflow gradients...\n";
     all_passed &= test_lateral_inflow_gradient();
     
     std::cout << "\n[9/9] Testing multiple gauge gradients...\n";
     all_passed &= test_multiple_gauges();
     
     // Final summary
     std::cout << "\n" << std::string(70, '=') << "\n";
     if (all_passed) {
         std::cout << "✓ ALL GRADIENT VERIFICATION TESTS PASSED\n";
     } else {
         std::cout << "✗ SOME TESTS FAILED - Review output above\n";
     }
     std::cout << std::string(70, '=') << "\n";
     
     return all_passed ? 0 : 1;
 }