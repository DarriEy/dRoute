/**
 * @file bindings.cpp
 * @brief Python bindings for dMC-Route using pybind11
 * 
 * Exposes the C++ routing library to Python for efficient integration
 * with PyTorch and other ML frameworks. Eliminates subprocess/CSV overhead.
 * 
 * Usage:
 *   import pydmc_route as dmc
 *   
 *   net = dmc.Network()
 *   net.load_from_geojson("network.geojson")
 *   
 *   config = dmc.RouterConfig()
 *   config.dt = 3600.0
 *   config.enable_gradients = True
 *   
 *   router = dmc.MuskingumCungeRouter(net, config)
 *   router.start_recording()
 *   
 *   for t in range(num_timesteps):
 *       router.set_lateral_inflows(runoff[t, :])
 *       router.route_timestep()
 *   
 *   router.stop_recording()
 *   router.compute_gradients([outlet_id], [1.0])
 *   grads = router.get_gradients()
 */

 #include <pybind11/pybind11.h>
 #include <pybind11/stl.h>
 #include <pybind11/numpy.h>
 #include <pybind11/functional.h>
 
 #include <dmc/router.hpp>
 #include <dmc/advanced_routing.hpp>
 #include <dmc/network.hpp>
 #include <dmc/network_io.hpp>
 
 namespace py = pybind11;
 using namespace dmc;
 
 // Helper to convert std::vector to numpy array
 template<typename T>
 py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
     return py::array_t<T>(vec.size(), vec.data());
 }
 
 // Helper to get Real value as double
 inline double get_real_value(const Real& r) {
     return to_double(r);
 }
 
 // Helper to set Real value from double
 inline void set_real_value(Real& r, double v) {
     r = Real(v);
 }
 
 // Wrapper function for loading network from GeoJSON
 inline Network load_network_geojson(const std::string& filepath) {
     NetworkIO io;
     return io.load_geojson(filepath);
 }
 
 // Wrapper function for loading network from CSV
 inline Network load_network_csv(const std::string& reaches_csv, 
                                  const std::string& params_csv = "") {
     NetworkIO io;
     return io.load_csv(reaches_csv, params_csv);
 }
 
 PYBIND11_MODULE(pydmc_route, m) {
     m.doc() = R"pbdoc(
         dMC-Route: Differentiable River Routing Library
         ------------------------------------------------
         
         A high-performance, differentiable routing library for
         hydrological modeling and machine learning applications.
         
         Features:
         - Multiple routing methods (Muskingum-Cunge, IRF, Diffusive Wave, KWT)
         - Automatic differentiation via CoDiPack
         - Direct memory access (no CSV/subprocess overhead)
         - NumPy array support for efficient data transfer
         
         Example:
             >>> import pydmc_route as dmc
             >>> net = dmc.Network()
             >>> config = dmc.RouterConfig()
             >>> router = dmc.MuskingumCungeRouter(net, config)
     )pbdoc";
 
     // =========================================================================
     // RouterConfig
     // =========================================================================
     py::class_<RouterConfig>(m, "RouterConfig", "Configuration for routing methods")
         .def(py::init<>())
         
         // Basic settings
         .def_readwrite("dt", &RouterConfig::dt,
             "Timestep in seconds (default: 3600.0)")
         .def_readwrite("enable_gradients", &RouterConfig::enable_gradients,
             "Enable gradient computation via AD (default: True)")
         .def_readwrite("min_flow", &RouterConfig::min_flow,
             "Minimum flow for numerical stability (default: 1e-6)")
         .def_readwrite("x_lower_bound", &RouterConfig::x_lower_bound,
             "Lower bound for Muskingum X parameter (default: 0.0)")
         .def_readwrite("x_upper_bound", &RouterConfig::x_upper_bound,
             "Upper bound for Muskingum X parameter (default: 0.5)")
         
         // AD smoothing
         .def_readwrite("use_smooth_bounds", &RouterConfig::use_smooth_bounds,
             "Use smooth min/max/clamp for AD (default: True)")
         .def_readwrite("smooth_epsilon", &RouterConfig::smooth_epsilon,
             "Epsilon for smooth functions (default: 1e-6)")
         
         // Sub-stepping
         .def_readwrite("fixed_substepping", &RouterConfig::fixed_substepping,
             "Use fixed substep count for AD safety (default: True)")
         .def_readwrite("num_substeps", &RouterConfig::num_substeps,
             "Number of substeps per reach (default: 4)")
         .def_readwrite("adaptive_substepping", &RouterConfig::adaptive_substepping,
             "Use adaptive substeps - breaks AD (default: False)")
         .def_readwrite("max_substeps", &RouterConfig::max_substeps,
             "Maximum substeps for adaptive mode (default: 10)")
         
         // IRF options
         .def_readwrite("irf_max_kernel_size", &RouterConfig::irf_max_kernel_size,
             "Maximum IRF kernel length in timesteps (default: 200)")
         .def_readwrite("irf_shape_param", &RouterConfig::irf_shape_param,
             "Gamma shape parameter for IRF (default: 2.5)")
         .def_readwrite("irf_soft_mask", &RouterConfig::irf_soft_mask,
             "Use sigmoid masking for differentiable IRF (default: True)")
         .def_readwrite("irf_mask_steepness", &RouterConfig::irf_mask_steepness,
             "Steepness of IRF sigmoid mask (default: 10.0)")
         
         // Diffusive wave options
         .def_readwrite("dw_num_nodes", &RouterConfig::dw_num_nodes,
             "Spatial nodes per reach for diffusive wave (default: 10)")
         .def_readwrite("dw_use_ift_adjoint", &RouterConfig::dw_use_ift_adjoint,
             "Use IFT for diffusive wave adjoints (default: True)")
         
         // KWT-Soft options
         .def_readwrite("kwt_gate_steepness", &RouterConfig::kwt_gate_steepness,
             "Soft gate steepness for KWT (default: 5.0)")
         .def_readwrite("kwt_anneal_steepness", &RouterConfig::kwt_anneal_steepness,
             "Enable steepness annealing during training (default: False)")
         .def_readwrite("kwt_steepness_min", &RouterConfig::kwt_steepness_min,
             "Minimum steepness for annealing (default: 1.0)")
         .def_readwrite("kwt_steepness_max", &RouterConfig::kwt_steepness_max,
             "Maximum steepness for annealing (default: 20.0)")
         
         // Memory management
         .def_readwrite("enable_checkpointing", &RouterConfig::enable_checkpointing,
             "Enable Revolve checkpointing (default: False)")
         .def_readwrite("checkpoint_interval", &RouterConfig::checkpoint_interval,
             "Timesteps between checkpoints (default: 1000)")
         
         // Parallelization
         .def_readwrite("parallel_routing", &RouterConfig::parallel_routing,
             "Enable OpenMP parallel routing (default: False)")
         .def_readwrite("num_threads", &RouterConfig::num_threads,
             "Thread count for parallel mode (default: 4)")
         
         .def("__repr__", [](const RouterConfig& c) {
             return "<RouterConfig dt=" + std::to_string(c.dt) + 
                    " enable_gradients=" + (c.enable_gradients ? "True" : "False") + ">";
         });
 
     // =========================================================================
     // ChannelGeometry
     // =========================================================================
     py::class_<ChannelGeometry>(m, "ChannelGeometry", "Channel geometry parameters")
         .def(py::init<>())
         .def_property("width_coef",
             [](const ChannelGeometry& g) { return to_double(g.width_coef); },
             [](ChannelGeometry& g, double v) { g.width_coef = Real(v); },
             "Width coefficient in W = a * Q^b")
         .def_property("width_exp",
             [](const ChannelGeometry& g) { return to_double(g.width_exp); },
             [](ChannelGeometry& g, double v) { g.width_exp = Real(v); },
             "Width exponent in W = a * Q^b")
         .def_property("depth_coef",
             [](const ChannelGeometry& g) { return to_double(g.depth_coef); },
             [](ChannelGeometry& g, double v) { g.depth_coef = Real(v); },
             "Depth coefficient in D = c * Q^d")
         .def_property("depth_exp",
             [](const ChannelGeometry& g) { return to_double(g.depth_exp); },
             [](ChannelGeometry& g, double v) { g.depth_exp = Real(v); },
             "Depth exponent in D = c * Q^d")
         .def("width", [](const ChannelGeometry& g, double Q) {
             return to_double(g.width(Real(Q)));
         }, "Compute channel width for given discharge")
         .def("depth", [](const ChannelGeometry& g, double Q) {
             return to_double(g.depth(Real(Q)));
         }, "Compute channel depth for given discharge");
 
     // =========================================================================
     // Reach
     // =========================================================================
     py::class_<Reach>(m, "Reach", "River reach segment")
         .def(py::init<>())
         .def_readwrite("id", &Reach::id)
         .def_readwrite("name", &Reach::name)
         .def_readwrite("length", &Reach::length, "Reach length in meters")
         .def_readwrite("slope", &Reach::slope, "Channel slope (m/m)")
         .def_readwrite("geometry", &Reach::geometry, "Channel geometry")
         .def_readwrite("upstream_junction_id", &Reach::upstream_junction_id)
         .def_readwrite("downstream_junction_id", &Reach::downstream_junction_id)
         
         // Manning's n as property (Real -> double conversion)
         .def_property("manning_n",
             [](const Reach& r) { return to_double(r.manning_n); },
             [](Reach& r, double v) { r.manning_n = Real(v); },
             "Manning's roughness coefficient")
         
         // Lateral inflow
         .def_property("lateral_inflow",
             [](const Reach& r) { return to_double(r.lateral_inflow); },
             [](Reach& r, double v) { r.lateral_inflow = Real(v); },
             "Lateral inflow rate (m³/s)")
         
         // Current state (read-only)
         .def_property_readonly("inflow",
             [](const Reach& r) { return to_double(r.inflow_curr); },
             "Current inflow (m³/s)")
         .def_property_readonly("outflow",
             [](const Reach& r) { return to_double(r.outflow_curr); },
             "Current outflow (m³/s)")
         
         // Gradients (read-only) - use lambdas for member access
         .def_property_readonly("grad_manning_n", 
             [](const Reach& r) { return r.grad_manning_n; },
             "Gradient w.r.t. Manning's n")
         .def_property_readonly("grad_width_coef",
             [](const Reach& r) { return r.grad_width_coef; },
             "Gradient w.r.t. width coefficient")
         .def_property_readonly("grad_width_exp",
             [](const Reach& r) { return r.grad_width_exp; },
             "Gradient w.r.t. width exponent")
         .def_property_readonly("grad_depth_coef",
             [](const Reach& r) { return r.grad_depth_coef; },
             "Gradient w.r.t. depth coefficient")
         .def_property_readonly("grad_depth_exp",
             [](const Reach& r) { return r.grad_depth_exp; },
             "Gradient w.r.t. depth exponent");
 
     // =========================================================================
     // Junction
     // =========================================================================
     py::class_<Junction>(m, "Junction", "River junction (confluence/bifurcation)")
         .def(py::init<>())
         .def_readwrite("id", &Junction::id)
         .def_readwrite("name", &Junction::name)
         .def_readwrite("upstream_reach_ids", &Junction::upstream_reach_ids)
         .def_readwrite("downstream_reach_ids", &Junction::downstream_reach_ids)
         .def_property("external_inflow",
             [](const Junction& j) { return to_double(j.external_inflow); },
             [](Junction& j, double v) { j.external_inflow = Real(v); },
             "External inflow at junction (m³/s)");
 
     // =========================================================================
     // Network
     // =========================================================================
     py::class_<Network>(m, "Network", "River network topology")
         .def(py::init<>())
         
         // Building network
         .def("add_reach", &Network::add_reach, "Add a reach to the network")
         .def("add_junction", &Network::add_junction, "Add a junction to the network")
         .def("build_topology", &Network::build_topology, "Build network topology")
         
         // Accessors
         .def("get_reach", py::overload_cast<int>(&Network::get_reach),
             py::return_value_policy::reference, "Get reach by ID")
         .def("get_junction", py::overload_cast<int>(&Network::get_junction),
             py::return_value_policy::reference, "Get junction by ID")
         .def("topological_order", &Network::topological_order,
             "Get reach IDs in topological (upstream-to-downstream) order")
         .def("num_reaches", &Network::num_reaches, "Number of reaches")
         .def("num_junctions", &Network::num_junctions, "Number of junctions")
         
         // Gradients
         .def("zero_gradients", &Network::zero_gradients, "Reset all gradients to zero")
         
         // Bulk operations for efficiency
         .def("set_manning_n_all", [](Network& net, py::array_t<double> values) {
             auto buf = values.request();
             double* ptr = static_cast<double*>(buf.ptr);
             const auto& order = net.topological_order();
             for (size_t i = 0; i < order.size() && i < static_cast<size_t>(buf.size); ++i) {
                 net.get_reach(order[i]).manning_n = Real(ptr[i]);
             }
         }, "Set Manning's n for all reaches (in topological order)")
         
         .def("get_manning_n_all", [](Network& net) {
             const auto& order = net.topological_order();
             std::vector<double> values(order.size());
             for (size_t i = 0; i < order.size(); ++i) {
                 values[i] = to_double(net.get_reach(order[i]).manning_n);
             }
             return vector_to_numpy(values);
         }, "Get Manning's n for all reaches (in topological order)")
         
         .def("get_grad_manning_n_all", [](Network& net) {
             const auto& order = net.topological_order();
             std::vector<double> grads(order.size());
             for (size_t i = 0; i < order.size(); ++i) {
                 grads[i] = net.get_reach(order[i]).grad_manning_n;
             }
             return vector_to_numpy(grads);
         }, "Get gradients w.r.t. Manning's n for all reaches")
         
         .def("set_lateral_inflows", [](Network& net, py::array_t<double> values) {
             auto buf = values.request();
             double* ptr = static_cast<double*>(buf.ptr);
             const auto& order = net.topological_order();
             for (size_t i = 0; i < order.size() && i < static_cast<size_t>(buf.size); ++i) {
                 net.get_reach(order[i]).lateral_inflow = Real(ptr[i]);
             }
         }, "Set lateral inflows for all reaches (in topological order)")
         
         .def("get_outflows", [](Network& net) {
             const auto& order = net.topological_order();
             std::vector<double> outflows(order.size());
             for (size_t i = 0; i < order.size(); ++i) {
                 outflows[i] = to_double(net.get_reach(order[i]).outflow_curr);
             }
             return vector_to_numpy(outflows);
         }, "Get current outflows for all reaches (in topological order)");
 
     // =========================================================================
     // Network I/O
     // =========================================================================
     m.def("load_network_geojson", &load_network_geojson,
         "Load network from GeoJSON file",
         py::arg("filepath"));
     
     m.def("load_network_csv", &load_network_csv,
         "Load network from CSV file",
         py::arg("reaches_csv"),
         py::arg("params_csv") = "");
     
     py::class_<NetworkLoadConfig>(m, "NetworkLoadConfig",
         "Configuration for network loading")
         .def(py::init<>())
         .def_readwrite("reach_id_prop", &NetworkLoadConfig::reach_id_prop)
         .def_readwrite("reach_length_prop", &NetworkLoadConfig::reach_length_prop)
         .def_readwrite("reach_slope_prop", &NetworkLoadConfig::reach_slope_prop)
         .def_readwrite("from_node_prop", &NetworkLoadConfig::from_node_prop)
         .def_readwrite("to_node_prop", &NetworkLoadConfig::to_node_prop)
         .def_readwrite("manning_n_prop", &NetworkLoadConfig::manning_n_prop)
         .def_readwrite("length_to_meters", &NetworkLoadConfig::length_to_meters)
         .def_readwrite("slope_factor", &NetworkLoadConfig::slope_factor)
         .def_readwrite("default_slope", &NetworkLoadConfig::default_slope)
         .def_readwrite("default_manning_n", &NetworkLoadConfig::default_manning_n)
         .def_readwrite("default_length", &NetworkLoadConfig::default_length)
         .def_readwrite("min_slope", &NetworkLoadConfig::min_slope);
     
     py::class_<NetworkIO>(m, "NetworkIO",
         "Network I/O helper class")
         .def(py::init<NetworkLoadConfig>(), py::arg("config") = NetworkLoadConfig())
         .def("load_geojson", &NetworkIO::load_geojson, py::arg("filepath"))
         .def("load_csv", &NetworkIO::load_csv, 
             py::arg("reaches_csv"), py::arg("params_csv") = "")
         .def("save_geojson", &NetworkIO::save_geojson,
             py::arg("network"), py::arg("filepath"))
         .def("save_csv", &NetworkIO::save_csv,
             py::arg("network"), py::arg("filepath"));
 
     // =========================================================================
     // MuskingumCungeRouter
     // =========================================================================
     py::class_<MuskingumCungeRouter>(m, "MuskingumCungeRouter",
         "Muskingum-Cunge routing with full AD support")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         
         // Core routing
         .def("route_timestep", &MuskingumCungeRouter::route_timestep,
             "Route one timestep")
         .def("route", &MuskingumCungeRouter::route, "Route multiple timesteps",
             py::arg("num_timesteps"))
         
         // Gradient computation
         .def("enable_gradients", &MuskingumCungeRouter::enable_gradients,
             "Enable/disable gradient computation")
         .def("start_recording", &MuskingumCungeRouter::start_recording,
             "Start recording for AD")
         .def("stop_recording", &MuskingumCungeRouter::stop_recording,
             "Stop recording for AD")
         .def("compute_gradients", &MuskingumCungeRouter::compute_gradients,
             "Compute gradients w.r.t. loss at gauge reaches",
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &MuskingumCungeRouter::get_gradients,
             "Get gradients as dict")
         .def("reset_gradients", &MuskingumCungeRouter::reset_gradients,
             "Reset gradients to zero")
         
         // State access
         .def("set_lateral_inflow", &MuskingumCungeRouter::set_lateral_inflow,
             "Set lateral inflow for a reach",
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &MuskingumCungeRouter::set_lateral_inflows,
             "Set lateral inflows for all reaches",
             py::arg("inflows"))
         .def("get_discharge", &MuskingumCungeRouter::get_discharge,
             "Get discharge at a reach",
             py::arg("reach_id"))
         .def("get_all_discharges", &MuskingumCungeRouter::get_all_discharges,
             "Get discharges at all reaches")
         .def("reset_state", &MuskingumCungeRouter::reset_state,
             "Reset router state")
         
         // Properties
         .def("current_time", &MuskingumCungeRouter::current_time,
             "Current simulation time")
         .def_property_readonly("network", 
             py::overload_cast<>(&MuskingumCungeRouter::network),
             py::return_value_policy::reference,
             "Access the network")
         .def_property_readonly("config", &MuskingumCungeRouter::config,
             "Access the configuration");
 
     // =========================================================================
     // IRFRouter
     // =========================================================================
     py::class_<IRFRouter>(m, "IRFRouter",
         "Impulse Response Function routing with soft-masked kernel")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &IRFRouter::route_timestep)
         .def("route", &IRFRouter::route, py::arg("num_timesteps"))
         .def("enable_gradients", &IRFRouter::enable_gradients)
         .def("start_recording", &IRFRouter::start_recording)
         .def("stop_recording", &IRFRouter::stop_recording)
         .def("compute_gradients", &IRFRouter::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &IRFRouter::get_gradients)
         .def("reset_gradients", &IRFRouter::reset_gradients)
         .def("set_lateral_inflow", &IRFRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &IRFRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &IRFRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &IRFRouter::get_all_discharges)
         .def("reset_state", &IRFRouter::reset_state)
         .def("current_time", &IRFRouter::current_time);
 
     // =========================================================================
     // DiffusiveWaveRouter
     // =========================================================================
     py::class_<DiffusiveWaveRouter>(m, "DiffusiveWaveRouter",
         "Diffusive wave routing with analytical gradients")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &DiffusiveWaveRouter::route_timestep)
         .def("route", &DiffusiveWaveRouter::route, py::arg("num_timesteps"))
         .def("enable_gradients", &DiffusiveWaveRouter::enable_gradients)
         .def("start_recording", &DiffusiveWaveRouter::start_recording)
         .def("stop_recording", &DiffusiveWaveRouter::stop_recording)
         .def("compute_gradients", &DiffusiveWaveRouter::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &DiffusiveWaveRouter::get_gradients)
         .def("reset_gradients", &DiffusiveWaveRouter::reset_gradients)
         .def("set_lateral_inflow", &DiffusiveWaveRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &DiffusiveWaveRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &DiffusiveWaveRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &DiffusiveWaveRouter::get_all_discharges)
         .def("reset_state", &DiffusiveWaveRouter::reset_state)
         .def("current_time", &DiffusiveWaveRouter::current_time);
 
     // =========================================================================
     // LagRouter
     // =========================================================================
     py::class_<LagRouter>(m, "LagRouter",
         "Simple lag routing (forward-only, not recommended for calibration)")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &LagRouter::route_timestep)
         .def("route", &LagRouter::route, py::arg("num_timesteps"))
         .def("enable_gradients", &LagRouter::enable_gradients)
         .def("start_recording", &LagRouter::start_recording)
         .def("stop_recording", &LagRouter::stop_recording)
         .def("compute_gradients", &LagRouter::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &LagRouter::get_gradients)
         .def("reset_gradients", &LagRouter::reset_gradients)
         .def("set_lateral_inflow", &LagRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &LagRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &LagRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &LagRouter::get_all_discharges)
         .def("reset_state", &LagRouter::reset_state)
         .def("current_time", &LagRouter::current_time);
 
     // =========================================================================
     // KWTRouter
     // =========================================================================
     py::class_<KWTRouter>(m, "KWTRouter",
         "Kinematic Wave Tracking (forward-only, mizuRoute compatible)")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &KWTRouter::route_timestep)
         .def("route", &KWTRouter::route, py::arg("num_timesteps"))
         .def("set_lateral_inflow", &KWTRouter::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", &KWTRouter::set_lateral_inflows,
             py::arg("inflows"))
         .def("get_discharge", &KWTRouter::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &KWTRouter::get_all_discharges)
         .def("reset_state", &KWTRouter::reset_state)
         .def("current_time", &KWTRouter::current_time);
 
     // =========================================================================
     // Advanced Routers (from advanced_routing.hpp)
     // =========================================================================
     py::class_<DiffusiveWaveIFT>(m, "DiffusiveWaveIFT",
         "Diffusive wave with IFT adjoint for exact gradients")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &DiffusiveWaveIFT::route_timestep)
         .def("route", &DiffusiveWaveIFT::route, py::arg("num_timesteps"))
         .def("enable_gradients", &DiffusiveWaveIFT::enable_gradients)
         .def("start_recording", &DiffusiveWaveIFT::start_recording)
         .def("stop_recording", &DiffusiveWaveIFT::stop_recording)
         .def("compute_gradients", &DiffusiveWaveIFT::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &DiffusiveWaveIFT::get_gradients)
         .def("reset_gradients", &DiffusiveWaveIFT::reset_gradients)
         .def("set_lateral_inflow", &DiffusiveWaveIFT::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", [](DiffusiveWaveIFT& router, const std::vector<std::pair<int, double>>& inflows) {
             for (const auto& [id, q] : inflows) {
                 router.set_lateral_inflow(id, q);
             }
         }, py::arg("inflows"), "Set lateral inflows as list of (reach_id, inflow) pairs")
         .def("get_discharge", &DiffusiveWaveIFT::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", [](DiffusiveWaveIFT& router) {
             std::vector<double> discharges;
             for (int id : router.network().topological_order()) {
                 discharges.push_back(router.get_discharge(id));
             }
             return discharges;
         }, "Get discharges for all reaches in topological order")
         .def("reset_state", &DiffusiveWaveIFT::reset_state)
         .def("current_time", &DiffusiveWaveIFT::current_time);
 
     py::class_<SoftGatedKWT>(m, "SoftGatedKWT",
         "Soft-gated KWT with differentiable parcel tracking")
         .def(py::init<Network&, RouterConfig>(),
             py::arg("network"), py::arg("config") = RouterConfig())
         .def("route_timestep", &SoftGatedKWT::route_timestep)
         .def("route", &SoftGatedKWT::route, py::arg("num_timesteps"))
         .def("enable_gradients", &SoftGatedKWT::enable_gradients)
         .def("start_recording", &SoftGatedKWT::start_recording)
         .def("stop_recording", &SoftGatedKWT::stop_recording)
         .def("compute_gradients", &SoftGatedKWT::compute_gradients,
             py::arg("gauge_reaches"), py::arg("dL_dQ"))
         .def("get_gradients", &SoftGatedKWT::get_gradients)
         .def("reset_gradients", &SoftGatedKWT::reset_gradients)
         .def("set_lateral_inflow", &SoftGatedKWT::set_lateral_inflow,
             py::arg("reach_id"), py::arg("inflow"))
         .def("set_lateral_inflows", [](SoftGatedKWT& router, const std::vector<std::pair<int, double>>& inflows) {
             for (const auto& [id, q] : inflows) {
                 router.set_lateral_inflow(id, q);
             }
         }, py::arg("inflows"), "Set lateral inflows as list of (reach_id, inflow) pairs")
         .def("get_discharge", &SoftGatedKWT::get_discharge, py::arg("reach_id"))
         .def("get_all_discharges", &SoftGatedKWT::get_all_discharges)
         .def("reset_state", &SoftGatedKWT::reset_state)
         .def("current_time", &SoftGatedKWT::current_time)
         .def("set_steepness", &SoftGatedKWT::set_steepness,
             "Set soft gate steepness (for annealing)")
         .def("get_steepness", &SoftGatedKWT::get_steepness,
             "Get current soft gate steepness");
 
     // =========================================================================
     // Version info
     // =========================================================================
     m.attr("__version__") = "0.5.0";
     m.attr("__author__") = "dMC-Route Authors";
 }