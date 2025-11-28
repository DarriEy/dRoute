/**
 * Shared library entry points for dMC-Route.
 * 
 * Provides C-compatible factory functions for creating BMI instances.
 */

#include <dmc/bmi.hpp>
#include <memory>

extern "C" {

/**
 * Create a new BMI model instance.
 * Caller is responsible for calling destroy_model().
 */
void* create_model() {
    return new dmc::BmiMuskingumCunge();
}

/**
 * Destroy a BMI model instance.
 */
void destroy_model(void* model) {
    delete static_cast<dmc::BmiMuskingumCunge*>(model);
}

/**
 * Get the BMI version.
 */
const char* get_bmi_version() {
    return "2.0";
}

/**
 * Check if AD is enabled.
 */
int is_ad_enabled() {
    return dmc::AD_ENABLED ? 1 : 0;
}

} // extern "C"
