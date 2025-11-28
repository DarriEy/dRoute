#ifndef DMC_ROUTE_BMI_HPP
#define DMC_ROUTE_BMI_HPP

#include "router.hpp"
#include <bmi.hxx>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>

namespace dmc {

/**
 * BMI adapter for MuskingumCungeRouter.
 * 
 * Provides the standard CSDMS BMI interface plus extensions
 * for differentiable coupling.
 */
class BmiMuskingumCunge : public bmi::Bmi {
public:
    BmiMuskingumCunge() = default;
    ~BmiMuskingumCunge() override = default;
    
    // ==================== Control Functions ====================
    
    void Initialize(std::string config_file) override;
    void Update() override;
    void UpdateUntil(double time) override;
    void Finalize() override;
    
    // ==================== Model Information ====================
    
    std::string GetComponentName() override {
        return "dMC-Route: Differentiable Muskingum-Cunge Router";
    }
    
    int GetInputItemCount() override {
        return static_cast<int>(input_var_names_.size());
    }
    
    int GetOutputItemCount() override {
        return static_cast<int>(output_var_names_.size());
    }
    
    std::vector<std::string> GetInputVarNames() override {
        return input_var_names_;
    }
    
    std::vector<std::string> GetOutputVarNames() override {
        return output_var_names_;
    }
    
    // ==================== Variable Information ====================
    
    int GetVarGrid(std::string name) override;
    std::string GetVarType(std::string name) override;
    std::string GetVarUnits(std::string name) override;
    int GetVarItemsize(std::string name) override;
    int GetVarNbytes(std::string name) override;
    std::string GetVarLocation(std::string name) override;
    
    // ==================== Time Information ====================
    
    double GetCurrentTime() override {
        return router_ ? router_->current_time() : 0.0;
    }
    
    double GetStartTime() override { return 0.0; }
    
    double GetEndTime() override { return end_time_; }
    
    std::string GetTimeUnits() override { return "s"; }
    
    double GetTimeStep() override {
        return router_ ? router_->config().dt : 3600.0;
    }
    
    // ==================== Getters and Setters ====================
    
    void GetValue(std::string name, void* dest) override;
    void* GetValuePtr(std::string name) override;
    void GetValueAtIndices(std::string name, void* dest, int* inds, int count) override;
    
    void SetValue(std::string name, void* src) override;
    void SetValueAtIndices(std::string name, int* inds, int count, void* src) override;
    
    // ==================== Grid Information ====================
    
    int GetGridRank(int grid) override { return 1; }
    int GetGridSize(int grid) override;
    std::string GetGridType(int grid) override { return "unstructured"; }
    
    void GetGridShape(int grid, int* shape) override;
    void GetGridSpacing(int grid, double* spacing) override;
    void GetGridOrigin(int grid, double* origin) override;
    
    void GetGridX(int grid, double* x) override;
    void GetGridY(int grid, double* y) override;
    void GetGridZ(int grid, double* z) override;
    
    int GetGridNodeCount(int grid) override;
    int GetGridEdgeCount(int grid) override;
    int GetGridFaceCount(int grid) override { return 0; }
    
    void GetGridEdgeNodes(int grid, int* edge_nodes) override;
    void GetGridFaceEdges(int grid, int* face_edges) override {}
    void GetGridFaceNodes(int grid, int* face_nodes) override {}
    void GetGridNodesPerFace(int grid, int* nodes_per_face) override {}
    
    // ==================== Extended Interface for AD ====================
    
    /**
     * Get names of learnable parameters.
     */
    std::vector<std::string> GetParameterNames();
    
    /**
     * Get current parameter values.
     */
    void GetParameters(const std::string& name, double* values);
    
    /**
     * Set parameter values.
     */
    void SetParameters(const std::string& name, const double* values);
    
    /**
     * Enable/disable gradient computation.
     */
    void EnableGradients(bool enable);
    
    /**
     * Start recording for gradient computation.
     * Call before simulation loop.
     */
    void StartRecording();
    
    /**
     * Stop recording.
     */
    void StopRecording();
    
    /**
     * Set the gradient of loss w.r.t. outputs (seeding for backprop).
     */
    void SetOutputGradients(const std::string& var_name, const double* gradients);
    
    /**
     * Compute gradients via reverse AD.
     */
    void ComputeGradients();
    
    /**
     * Get gradient of loss w.r.t. a parameter.
     */
    void GetParameterGradients(const std::string& param_name, double* gradients);
    
    /**
     * Reset gradients and tape.
     */
    void ResetGradients();
    
    // ==================== Direct Access ====================
    
    MuskingumCungeRouter* get_router() { return router_.get(); }
    Network* get_network() { return network_.get(); }
    
private:
    std::unique_ptr<Network> network_;
    std::unique_ptr<MuskingumCungeRouter> router_;
    
    double end_time_ = 0.0;
    
    std::vector<std::string> input_var_names_ = {
        "lateral_inflow",       // Runoff from catchments [m³/s]
        "upstream_boundary"     // Boundary inflow [m³/s]
    };
    
    std::vector<std::string> output_var_names_ = {
        "discharge",            // Flow at each reach [m³/s]
        "velocity",             // Flow velocity [m/s]
        "depth",                // Water depth [m]
        "storage"               // Channel storage [m³]
    };
    
    // Gradient seeding storage
    std::unordered_map<int, double> output_gradients_;
    std::vector<int> gauge_reaches_;
    
    // Internal buffers
    std::vector<double> lateral_inflow_buffer_;
    std::vector<double> discharge_buffer_;
    
    void load_config(const std::string& config_file);
    void build_variable_maps();
};

// ==================== Implementation ====================

inline void BmiMuskingumCunge::Initialize(std::string config_file) {
    load_config(config_file);
    
    // Allocate buffers
    size_t n = network_->num_reaches();
    lateral_inflow_buffer_.resize(n, 0.0);
    discharge_buffer_.resize(n, 0.0);
    
    build_variable_maps();
}

inline void BmiMuskingumCunge::Update() {
    if (!router_) return;
    router_->route_timestep();
}

inline void BmiMuskingumCunge::UpdateUntil(double time) {
    if (!router_) return;
    while (router_->current_time() < time) {
        router_->route_timestep();
    }
}

inline void BmiMuskingumCunge::Finalize() {
    router_.reset();
    network_.reset();
}

inline int BmiMuskingumCunge::GetVarGrid(std::string name) {
    // All variables use grid 0 (the reach network)
    return 0;
}

inline std::string BmiMuskingumCunge::GetVarType(std::string name) {
    return "double";
}

inline std::string BmiMuskingumCunge::GetVarUnits(std::string name) {
    if (name == "lateral_inflow" || name == "upstream_boundary" || name == "discharge") {
        return "m^3 s^-1";
    } else if (name == "velocity") {
        return "m s^-1";
    } else if (name == "depth") {
        return "m";
    } else if (name == "storage") {
        return "m^3";
    }
    return "";
}

inline int BmiMuskingumCunge::GetVarItemsize(std::string name) {
    return sizeof(double);
}

inline int BmiMuskingumCunge::GetVarNbytes(std::string name) {
    return static_cast<int>(network_->num_reaches() * sizeof(double));
}

inline std::string BmiMuskingumCunge::GetVarLocation(std::string name) {
    return "edge";  // Variables defined on reaches (edges)
}

inline void BmiMuskingumCunge::GetValue(std::string name, void* dest) {
    double* ptr = static_cast<double*>(dest);
    const auto& order = network_->topological_order();
    
    if (name == "discharge") {
        for (size_t i = 0; i < order.size(); ++i) {
            ptr[i] = router_->get_discharge(order[i]);
        }
    } else if (name == "velocity") {
        for (size_t i = 0; i < order.size(); ++i) {
            ptr[i] = to_double(network_->get_reach(order[i]).velocity);
        }
    } else if (name == "depth") {
        for (size_t i = 0; i < order.size(); ++i) {
            const Reach& r = network_->get_reach(order[i]);
            ptr[i] = to_double(r.geometry.depth(r.outflow_curr));
        }
    }
}

inline void* BmiMuskingumCunge::GetValuePtr(std::string name) {
    // Copy to buffer and return pointer
    // Note: This is not ideal for performance but matches BMI spec
    GetValue(name, discharge_buffer_.data());
    return discharge_buffer_.data();
}

inline void BmiMuskingumCunge::GetValueAtIndices(std::string name, void* dest, 
                                                   int* inds, int count) {
    double* ptr = static_cast<double*>(dest);
    const auto& order = network_->topological_order();
    
    if (name == "discharge") {
        for (int i = 0; i < count; ++i) {
            ptr[i] = router_->get_discharge(order[inds[i]]);
        }
    }
}

inline void BmiMuskingumCunge::SetValue(std::string name, void* src) {
    const double* ptr = static_cast<const double*>(src);
    const auto& order = network_->topological_order();
    
    if (name == "lateral_inflow") {
        for (size_t i = 0; i < order.size(); ++i) {
            router_->set_lateral_inflow(order[i], ptr[i]);
        }
    }
}

inline void BmiMuskingumCunge::SetValueAtIndices(std::string name, int* inds, 
                                                   int count, void* src) {
    const double* ptr = static_cast<const double*>(src);
    const auto& order = network_->topological_order();
    
    if (name == "lateral_inflow") {
        for (int i = 0; i < count; ++i) {
            router_->set_lateral_inflow(order[inds[i]], ptr[i]);
        }
    }
}

inline int BmiMuskingumCunge::GetGridSize(int grid) {
    return static_cast<int>(network_->num_reaches());
}

inline void BmiMuskingumCunge::GetGridShape(int grid, int* shape) {
    shape[0] = static_cast<int>(network_->num_reaches());
}

inline void BmiMuskingumCunge::GetGridSpacing(int grid, double* spacing) {
    // N/A for unstructured
}

inline void BmiMuskingumCunge::GetGridOrigin(int grid, double* origin) {
    // N/A for unstructured
}

inline void BmiMuskingumCunge::GetGridX(int grid, double* x) {
    // Would need coordinates stored in network
}

inline void BmiMuskingumCunge::GetGridY(int grid, double* y) {
    // Would need coordinates stored in network
}

inline void BmiMuskingumCunge::GetGridZ(int grid, double* z) {
    // Would need coordinates stored in network
}

inline int BmiMuskingumCunge::GetGridNodeCount(int grid) {
    return static_cast<int>(network_->num_junctions());
}

inline int BmiMuskingumCunge::GetGridEdgeCount(int grid) {
    return static_cast<int>(network_->num_reaches());
}

inline void BmiMuskingumCunge::GetGridEdgeNodes(int grid, int* edge_nodes) {
    // Would fill with [upstream_junction, downstream_junction] pairs
}

// ==================== Extended AD Interface ====================

inline std::vector<std::string> BmiMuskingumCunge::GetParameterNames() {
    return network_->get_parameter_names();
}

inline void BmiMuskingumCunge::EnableGradients(bool enable) {
    router_->enable_gradients(enable);
}

inline void BmiMuskingumCunge::StartRecording() {
    router_->start_recording();
}

inline void BmiMuskingumCunge::StopRecording() {
    router_->stop_recording();
}

inline void BmiMuskingumCunge::SetOutputGradients(const std::string& var_name, 
                                                    const double* gradients) {
    const auto& order = network_->topological_order();
    for (size_t i = 0; i < order.size(); ++i) {
        output_gradients_[order[i]] = gradients[i];
        if (std::abs(gradients[i]) > 1e-10) {
            gauge_reaches_.push_back(order[i]);
        }
    }
}

inline void BmiMuskingumCunge::ComputeGradients() {
    std::vector<double> dL_dQ;
    for (int reach_id : gauge_reaches_) {
        dL_dQ.push_back(output_gradients_[reach_id]);
    }
    router_->compute_gradients(gauge_reaches_, dL_dQ);
}

inline void BmiMuskingumCunge::GetParameterGradients(const std::string& param_name, 
                                                       double* gradients) {
    auto all_grads = router_->get_gradients();
    const auto& order = network_->topological_order();
    
    // Parse parameter type from name (e.g., "manning_n")
    for (size_t i = 0; i < order.size(); ++i) {
        std::string key = "reach_" + std::to_string(order[i]) + "_" + param_name;
        if (all_grads.count(key)) {
            gradients[i] = all_grads[key];
        } else {
            gradients[i] = 0.0;
        }
    }
}

inline void BmiMuskingumCunge::ResetGradients() {
    router_->reset_gradients();
    output_gradients_.clear();
    gauge_reaches_.clear();
}

inline void BmiMuskingumCunge::load_config(const std::string& config_file) {
    // Minimal YAML-like parsing (replace with yaml-cpp in production)
    // For now, create a simple test network
    
    network_ = std::make_unique<Network>();
    
    // Example: create a simple chain of 3 reaches
    for (int i = 0; i < 3; ++i) {
        Reach r;
        r.id = i;
        r.name = "reach_" + std::to_string(i);
        r.length = 5000.0;  // 5 km
        r.slope = 0.001;    // 0.1%
        r.manning_n = Real(0.035);
        r.upstream_junction_id = (i > 0) ? i - 1 : -1;
        r.downstream_junction_id = i;
        network_->add_reach(r);
        
        Junction j;
        j.id = i;
        if (i > 0) {
            j.upstream_reach_ids = {i - 1};
        }
        if (i < 2) {
            j.downstream_reach_ids = {i + 1};
        }
        j.is_headwater = (i == 0);
        j.is_outlet = (i == 2);
        network_->add_junction(j);
    }
    
    network_->build_topology();
    
    RouterConfig config;
    config.dt = 3600.0;
    config.enable_gradients = true;
    
    router_ = std::make_unique<MuskingumCungeRouter>(*network_, config);
    
    end_time_ = 86400.0 * 365;  // 1 year
}

inline void BmiMuskingumCunge::build_variable_maps() {
    // Setup internal mappings
}

} // namespace dmc

#endif // DMC_ROUTE_BMI_HPP
