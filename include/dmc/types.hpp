#ifndef DMC_ROUTE_TYPES_HPP
#define DMC_ROUTE_TYPES_HPP

// Configuration-based AD type selection
#ifdef DMC_USE_CODIPACK
    #include <codi.hpp>
    
    namespace dmc {
        // Primary type for gradient computation
        using Real = codi::RealReverse;
        using Tape = typename Real::Tape;
        
        // Alternative types for different use cases
        using RealForward = codi::RealForward;       // Forward mode (few params)
        using RealReverseIdx = codi::RealReverseIndex; // Memory-efficient reverse
        
        constexpr bool AD_ENABLED = true;
        
        inline Tape& get_tape() {
            return Real::getTape();
        }
        
        inline void activate_tape() {
            get_tape().setActive();
        }
        
        inline void deactivate_tape() {
            get_tape().setPassive();
        }
        
        inline void reset_tape() {
            get_tape().reset();
        }
        
        template<typename T>
        inline void register_input(T& val) {
            get_tape().registerInput(val);
        }
        
        template<typename T>
        inline void register_output(T& val) {
            get_tape().registerOutput(val);
        }
        
        template<typename T>
        inline void set_gradient(T& val, double grad) {
            val.setGradient(grad);
        }
        
        template<typename T>
        inline double get_gradient(const T& val) {
            return val.getGradient();
        }
        
        inline void evaluate_tape() {
            get_tape().evaluate();
        }
        
        template<typename T>
        inline double to_double(const T& val) {
            return codi::RealTraits::getPassiveValue(val);
        }
    }
    
#else
    // No AD - use plain doubles
    namespace dmc {
        using Real = double;
        
        constexpr bool AD_ENABLED = false;
        
        inline void activate_tape() {}
        inline void deactivate_tape() {}
        inline void reset_tape() {}
        
        template<typename T>
        inline void register_input(T&) {}
        
        template<typename T>
        inline void register_output(T&) {}
        
        template<typename T>
        inline void set_gradient(T&, double) {}
        
        template<typename T>
        inline double get_gradient(const T&) { return 0.0; }
        
        inline void evaluate_tape() {}
        
        template<typename T>
        inline double to_double(const T& val) { return val; }
    }
#endif

#include <cmath>
#include <algorithm>

namespace dmc {
    // Safe math operations that work with both AD and non-AD types
    template<typename T>
    inline T safe_pow(const T& base, double exp) {
        using std::pow;
        return pow(base, exp);
    }
    
    // AD-safe power with Real exponent: x^y = exp(y * log(x))
    template<typename T>
    inline T safe_pow_real(const T& base, const T& exponent) {
        using std::log;
        T safe_base = (base > T(1e-10)) ? base : T(1e-10);
        return exp(exponent * log(safe_base));
    }
    
    template<typename T>
    inline T safe_sqrt(const T& x) {
        using std::sqrt;
        return sqrt(x);
    }
    
    /**
     * Smooth maximum for AD (softmax approximation)
     * 
     * Uses: smooth_max(a, b) ≈ 0.5 * (a + b + sqrt((a-b)² + ε))
     * 
     * @param a First value
     * @param b Second value  
     * @param epsilon Smoothing parameter (smaller = sharper transition)
     */
    template<typename T>
    inline T smooth_max(const T& a, const T& b, double epsilon = 1e-6) {
        using std::sqrt;
        T diff = a - b;
        return T(0.5) * (a + b + sqrt(diff * diff + T(epsilon)));
    }
    
    /**
     * Smooth minimum for AD (softmin approximation)
     */
    template<typename T>
    inline T smooth_min(const T& a, const T& b, double epsilon = 1e-6) {
        using std::sqrt;
        T diff = a - b;
        return T(0.5) * (a + b - sqrt(diff * diff + T(epsilon)));
    }
    
    /**
     * Smooth clamp for AD
     * Ensures smooth gradients at boundaries
     */
    template<typename T>
    inline T smooth_clamp(const T& val, const T& lo, const T& hi, double epsilon = 1e-6) {
        return smooth_min(smooth_max(val, lo, epsilon), hi, epsilon);
    }
    
    /**
     * Hard max (standard, non-smooth)
     * Use when gradients at discontinuity don't matter
     */
    template<typename T>
    inline T safe_max(const T& a, const T& b) {
        return (a > b) ? a : b;
    }
    
    template<typename T>
    inline T safe_min(const T& a, const T& b) {
        return (a < b) ? a : b;
    }
    
    template<typename T>
    inline T clamp(const T& val, const T& lo, const T& hi) {
        return safe_max(lo, safe_min(val, hi));
    }
    
    /**
     * Sigmoid function for smooth transitions
     * σ(x) = 1 / (1 + exp(-x))
     */
    template<typename T>
    inline T sigmoid(const T& x) {
        using std::exp;
        double x_val = to_double(x);
        if (x_val > 20.0) return T(1.0);
        if (x_val < -20.0) return T(0.0);
        return T(1.0) / (T(1.0) + exp(-x));
    }
    
    /**
     * Soft step function
     * Smooth approximation to step(x - threshold)
     */
    template<typename T>
    inline T soft_step(const T& x, const T& threshold, double steepness = 10.0) {
        return sigmoid((x - threshold) * T(steepness));
    }
}

#endif // DMC_ROUTE_TYPES_HPP
