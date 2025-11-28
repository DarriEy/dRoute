/**
 * BMI interface tests.
 */

#include <dmc/bmi.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dmc;

void test_initialization() {
    std::cout << "Testing initialization... ";
    
    BmiMuskingumCunge model;
    model.Initialize("test_config.yaml");
    
    assert(model.GetComponentName().find("dMC-Route") != std::string::npos);
    assert(model.GetTimeStep() > 0);
    assert(model.GetInputVarNames().size() > 0);
    assert(model.GetOutputVarNames().size() > 0);
    
    model.Finalize();
    std::cout << "PASS\n";
}

void test_update() {
    std::cout << "Testing update... ";
    
    BmiMuskingumCunge model;
    model.Initialize("test_config.yaml");
    
    double t0 = model.GetCurrentTime();
    model.Update();
    double t1 = model.GetCurrentTime();
    
    assert(t1 > t0);
    assert(std::abs(t1 - t0 - model.GetTimeStep()) < 1e-6);
    
    model.Finalize();
    std::cout << "PASS\n";
}

void test_getset_value() {
    std::cout << "Testing get/set value... ";
    
    BmiMuskingumCunge model;
    model.Initialize("test_config.yaml");
    
    // Set lateral inflow
    int grid_size = model.GetGridSize(0);
    std::vector<double> inflow(grid_size, 5.0);
    model.SetValue("lateral_inflow", inflow.data());
    
    // Run a few steps
    for (int i = 0; i < 5; ++i) {
        model.Update();
    }
    
    // Get discharge
    std::vector<double> discharge(grid_size);
    model.GetValue("discharge", discharge.data());
    
    // Should have some positive discharge
    bool has_flow = false;
    for (double q : discharge) {
        if (q > 1e-6) has_flow = true;
    }
    assert(has_flow);
    
    model.Finalize();
    std::cout << "PASS\n";
}

void test_variable_info() {
    std::cout << "Testing variable info... ";
    
    BmiMuskingumCunge model;
    model.Initialize("test_config.yaml");
    
    assert(model.GetVarType("discharge") == "double");
    assert(model.GetVarUnits("discharge") == "m^3 s^-1");
    assert(model.GetVarItemsize("discharge") == sizeof(double));
    assert(model.GetVarGrid("discharge") == 0);
    
    model.Finalize();
    std::cout << "PASS\n";
}

void test_time_functions() {
    std::cout << "Testing time functions... ";
    
    BmiMuskingumCunge model;
    model.Initialize("test_config.yaml");
    
    assert(model.GetStartTime() == 0.0);
    assert(model.GetEndTime() > 0.0);
    assert(model.GetTimeUnits() == "s");
    
    double target_time = model.GetCurrentTime() + 5 * model.GetTimeStep();
    model.UpdateUntil(target_time);
    
    assert(model.GetCurrentTime() >= target_time);
    
    model.Finalize();
    std::cout << "PASS\n";
}

void test_grid_info() {
    std::cout << "Testing grid info... ";
    
    BmiMuskingumCunge model;
    model.Initialize("test_config.yaml");
    
    assert(model.GetGridRank(0) == 1);
    assert(model.GetGridSize(0) > 0);
    assert(model.GetGridType(0) == "unstructured");
    assert(model.GetGridNodeCount(0) >= 0);
    assert(model.GetGridEdgeCount(0) > 0);
    
    model.Finalize();
    std::cout << "PASS\n";
}

void test_extended_ad_interface() {
    std::cout << "Testing extended AD interface... ";
    
    BmiMuskingumCunge model;
    model.Initialize("test_config.yaml");
    
    // Get parameter names
    auto param_names = model.GetParameterNames();
    assert(param_names.size() > 0);
    
    // Check that manning_n is a parameter
    bool has_manning = false;
    for (const auto& name : param_names) {
        if (name.find("manning_n") != std::string::npos) {
            has_manning = true;
            break;
        }
    }
    assert(has_manning);
    
    if (AD_ENABLED) {
        // Enable gradients
        model.EnableGradients(true);
        model.StartRecording();
        
        // Set inflow and run
        int grid_size = model.GetGridSize(0);
        std::vector<double> inflow(grid_size, 2.0);
        model.SetValue("lateral_inflow", inflow.data());
        
        for (int i = 0; i < 3; ++i) {
            model.Update();
        }
        
        model.StopRecording();
        
        // Set output gradients
        std::vector<double> dL_dQ(grid_size, 0.0);
        dL_dQ.back() = 1.0;  // Gradient only at outlet
        model.SetOutputGradients("discharge", dL_dQ.data());
        
        // Compute gradients
        model.ComputeGradients();
        
        // Get parameter gradients
        std::vector<double> grad_n(grid_size);
        model.GetParameterGradients("manning_n", grad_n.data());
        
        // Reset
        model.ResetGradients();
    }
    
    model.Finalize();
    std::cout << "PASS\n";
}

int main() {
    std::cout << "=== BMI Interface Tests ===\n\n";
    
    try {
        test_initialization();
        test_update();
        test_getset_value();
        test_variable_info();
        test_time_functions();
        test_grid_info();
        test_extended_ad_interface();
        
        std::cout << "\n=== All BMI tests passed! ===\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nTest failed with exception: " << e.what() << "\n";
        return 1;
    }
}
