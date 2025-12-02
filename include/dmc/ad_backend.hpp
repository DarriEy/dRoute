/**
 * @file ad_backend.hpp
 * @brief Automatic Differentiation Backend Selection
 * 
 * Provides runtime dispatch between AD frameworks:
 * - CoDiPack: Operator-overloading, tape-based reverse-mode AD
 * - Enzyme: Source-to-source compiler transformation AD
 * - None: No differentiation (plain doubles, fastest forward pass)
 * 
 * This design allows:
 * 1. Keeping CoDiPack as validated reference implementation
 * 2. Adding Enzyme pathway for large-domain/GPU scenarios
 * 3. Runtime switching for benchmarking and validation
 */

#ifndef DMC_AD_BACKEND_HPP
#define DMC_AD_BACKEND_HPP

#include <string>
#include <stdexcept>

namespace dmc {

// ============================================================================
// AD Backend Enumeration
// ============================================================================

/**
 * @brief Available AD backends
 */
enum class ADBackend {
    NONE,       ///< No AD - plain double, forward-only
    CODIPACK,   ///< CoDiPack operator overloading (existing implementation)
    ENZYME      ///< Enzyme source transformation (new implementation)
};

/**
 * @brief Convert string to ADBackend
 */
inline ADBackend ad_backend_from_string(const std::string& s) {
    if (s == "none" || s == "NONE" || s == "None") return ADBackend::NONE;
    if (s == "codipack" || s == "CODIPACK" || s == "CoDiPack") return ADBackend::CODIPACK;
    if (s == "enzyme" || s == "ENZYME" || s == "Enzyme") return ADBackend::ENZYME;
    throw std::invalid_argument("Unknown AD backend: " + s);
}

/**
 * @brief Convert ADBackend to string
 */
inline std::string ad_backend_to_string(ADBackend backend) {
    switch (backend) {
        case ADBackend::NONE: return "none";
        case ADBackend::CODIPACK: return "codipack";
        case ADBackend::ENZYME: return "enzyme";
    }
    return "unknown";
}

// ============================================================================
// Compile-Time Feature Detection
// ============================================================================

#ifdef DMC_USE_CODIPACK
    constexpr bool CODIPACK_AVAILABLE = true;
#else
    constexpr bool CODIPACK_AVAILABLE = false;
#endif

#ifdef DMC_USE_ENZYME
    constexpr bool ENZYME_AVAILABLE = true;
#else
    constexpr bool ENZYME_AVAILABLE = false;
#endif

/**
 * @brief Check if a backend is available at compile time
 */
inline bool is_backend_available(ADBackend backend) {
    switch (backend) {
        case ADBackend::NONE: return true;
        case ADBackend::CODIPACK: return CODIPACK_AVAILABLE;
        case ADBackend::ENZYME: return ENZYME_AVAILABLE;
    }
    return false;
}

/**
 * @brief Get default backend based on compile-time availability
 */
inline ADBackend get_default_backend() {
    if (CODIPACK_AVAILABLE) return ADBackend::CODIPACK;
    if (ENZYME_AVAILABLE) return ADBackend::ENZYME;
    return ADBackend::NONE;
}

// ============================================================================
// Enzyme Declarations (when available)
// ============================================================================

#ifdef DMC_USE_ENZYME

extern "C" {
    void __enzyme_autodiff(void*, ...);
    int enzyme_dup;
    int enzyme_dupnoneed;
    int enzyme_const;
    int enzyme_out;
}

#define ENZYME_DUP enzyme_dup,
#define ENZYME_CONST enzyme_const,
#define ENZYME_DUPNONEED enzyme_dupnoneed,
#define ENZYME_OUT enzyme_out,

#else

// Stub macros when Enzyme not available
#define ENZYME_DUP
#define ENZYME_CONST
#define ENZYME_DUPNONEED
#define ENZYME_OUT

#endif // DMC_USE_ENZYME

// ============================================================================
// Host/Device Macros (for future CUDA support)
// ============================================================================

#ifdef __CUDACC__
    #define DMC_HOST __host__
    #define DMC_DEVICE __device__
    #define DMC_HOST_DEVICE __host__ __device__
#else
    #define DMC_HOST
    #define DMC_DEVICE
    #define DMC_HOST_DEVICE
#endif

// ============================================================================
// Gradient Result Structure (Backend-Agnostic)
// ============================================================================

/**
 * @brief Container for gradient results
 * 
 * Provides a unified interface for gradient storage regardless of
 * which AD backend computed them.
 */
struct GradientResult {
    // Per-reach gradients (indexed by reach position in topological order)
    std::vector<double> d_manning_n;
    std::vector<double> d_width_coef;
    std::vector<double> d_width_exp;
    std::vector<double> d_depth_coef;
    std::vector<double> d_depth_exp;
    
    // Metadata
    ADBackend backend = ADBackend::NONE;
    double loss = 0.0;
    bool valid = false;
    
    void resize(size_t n_reaches) {
        d_manning_n.resize(n_reaches, 0.0);
        d_width_coef.resize(n_reaches, 0.0);
        d_width_exp.resize(n_reaches, 0.0);
        d_depth_coef.resize(n_reaches, 0.0);
        d_depth_exp.resize(n_reaches, 0.0);
    }
    
    void zero() {
        std::fill(d_manning_n.begin(), d_manning_n.end(), 0.0);
        std::fill(d_width_coef.begin(), d_width_coef.end(), 0.0);
        std::fill(d_width_exp.begin(), d_width_exp.end(), 0.0);
        std::fill(d_depth_coef.begin(), d_depth_coef.end(), 0.0);
        std::fill(d_depth_exp.begin(), d_depth_exp.end(), 0.0);
        loss = 0.0;
        valid = false;
    }
};

} // namespace dmc

#endif // DMC_AD_BACKEND_HPP
