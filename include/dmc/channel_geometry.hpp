/**
 * @file channel_geometry.hpp
 * @brief Generalized hydraulic geometry for river channels
 * 
 * Implements multiple cross-section types with analytical derivatives
 * for use in differentiable routing. Supports:
 * - Power-law (Leopold-Maddock) geometry
 * - Trapezoidal channels
 * - Compound channels (main channel + floodplain)
 * 
 * All methods provide derivatives of hydraulic radius w.r.t. flow area
 * required for the kinematic wave celerity formula.
 */

#ifndef DMC_CHANNEL_GEOMETRY_HPP
#define DMC_CHANNEL_GEOMETRY_HPP

#include "types.hpp"
#include <cmath>
#include <algorithm>

namespace dmc {

/**
 * Channel cross-section type enumeration
 */
enum class ChannelType {
    POWER_LAW,      // Leopold-Maddock power laws (default)
    TRAPEZOIDAL,    // Trapezoidal with side slopes
    COMPOUND        // Main channel + floodplain
};

/**
 * Parameters for power-law (Leopold-Maddock) geometry
 * 
 * W = a * Q^b  (width)
 * D = c * Q^f  (depth)
 * V = k * Q^m  (velocity, derived)
 * 
 * Constraint: b + f + m = 1 (continuity)
 */
struct PowerLawParams {
    Real width_coef = Real(7.2);      // a coefficient
    Real width_exp = Real(0.5);       // b exponent
    Real depth_coef = Real(0.27);     // c coefficient
    Real depth_exp = Real(0.3);       // f exponent
    
    // Derived: velocity_exp = 1 - width_exp - depth_exp (m = 1 - b - f)
};

/**
 * Parameters for trapezoidal channel
 * 
 *     ____w_top____
 *    /             \
 *   /               \  side_slope (H:V)
 *  /___w_bottom______\
 *         |
 *       depth
 */
struct TrapezoidalParams {
    Real bottom_width = Real(10.0);   // Bottom width [m]
    Real side_slope = Real(2.0);      // Horizontal:Vertical (e.g., 2:1)
    Real max_depth = Real(5.0);       // Maximum depth [m]
};

/**
 * Parameters for compound channel (main channel + floodplain)
 * 
 *  _____________________________
 * |   floodplain   |   |   fp  |
 * |________________|   |_______|  <- bankfull_depth
 *                  |   |
 *                  |   |  <- main channel
 *                  |___|
 */
struct CompoundParams {
    // Main channel (trapezoidal)
    Real main_bottom_width = Real(15.0);
    Real main_side_slope = Real(1.5);
    Real bankfull_depth = Real(3.0);
    
    // Floodplain (rectangular, both sides)
    Real floodplain_width = Real(100.0);  // Total floodplain width
    Real floodplain_roughness_factor = Real(1.5);  // n_fp / n_main
};

/**
 * Unified channel geometry class
 * 
 * Computes hydraulic properties for any channel type with
 * support for automatic differentiation.
 */
class ChannelGeometry {
public:
    ChannelType type = ChannelType::POWER_LAW;
    
    // Type-specific parameters
    PowerLawParams power_law;
    TrapezoidalParams trapezoidal;
    CompoundParams compound;
    
    /**
     * Compute all hydraulic properties from discharge
     * 
     * @param Q Discharge [m³/s]
     * @param S Channel slope [-]
     * @param n Manning's roughness [-]
     * @return Tuple of (area, wetted_perimeter, hydraulic_radius, top_width, depth)
     */
    struct HydraulicProps {
        Real area;
        Real wetted_perimeter;
        Real hydraulic_radius;
        Real top_width;
        Real depth;
        Real celerity_factor;  // dQ/dA factor for kinematic wave celerity
    };
    
    HydraulicProps compute_properties(Real Q, double S, Real n) const {
        switch (type) {
            case ChannelType::POWER_LAW:
                return compute_power_law(Q, S, n);
            case ChannelType::TRAPEZOIDAL:
                return compute_trapezoidal(Q, S, n);
            case ChannelType::COMPOUND:
                return compute_compound(Q, S, n);
            default:
                return compute_power_law(Q, S, n);
        }
    }
    
    /**
     * Compute kinematic wave celerity with proper geometry
     * 
     * c = (1/B) * dQ/dy = (1/B) * dQ/dA * dA/dy = (1/B) * c_factor * B = c_factor
     * 
     * For power-law: c = (5/3 - 2b/3) * V  [generalized wide-channel formula]
     * For rectangular: c = 5/3 * V
     * For trapezoidal: c depends on shape
     */
    Real compute_celerity(Real Q, double S, Real n) const {
        auto props = compute_properties(Q, S, n);
        Real velocity = Q / props.area;
        return props.celerity_factor * velocity;
    }
    
    /**
     * Compute derivative of celerity w.r.t. Manning's n
     * 
     * dc/dn = dc/dv * dv/dn = c_factor * (-v/n) = -c/n
     */
    Real compute_dcelerity_dn(Real Q, double S, Real n) const {
        Real c = compute_celerity(Q, S, n);
        return -c / n;
    }

private:
    /**
     * Power-law (Leopold-Maddock) geometry
     * 
     * Generalizes beyond the "5/3 wide rectangular" assumption.
     * 
     * For a power-law channel:
     *   W = a * Q^b
     *   D = c * Q^f
     *   A = W * D = a*c * Q^(b+f)
     *   
     * The kinematic wave celerity is:
     *   c = dQ/dA = Q / A * 1/(b+f) = V / (b+f)
     *   
     * For wide rectangular (b=0.5, f=0.4), this gives c ≈ 1.11 * V
     * The classic "5/3 * V" assumes b=0.5, f=0.3 giving b+f=0.8, so c = V/0.8 = 1.25V
     * But the true 5/3 formula comes from Manning's equation applied to wide channels.
     * 
     * General formula: c = (5/3 - 2b/3) * V for power-law channels
     * This reduces to 5/3 * V when b=0 (rectangular).
     */
    HydraulicProps compute_power_law(Real Q, double S, Real n) const {
        HydraulicProps props;
        
        // Ensure minimum Q for stability
        Real Q_safe = safe_max(Q, Real(0.001));
        
        // Width and depth from power laws
        props.top_width = power_law.width_coef * pow(Q_safe, power_law.width_exp);
        props.depth = power_law.depth_coef * pow(Q_safe, power_law.depth_exp);
        
        // Ensure reasonable bounds
        props.top_width = safe_max(props.top_width, Real(0.5));
        props.depth = safe_max(props.depth, Real(0.05));
        
        // Area (rectangular approximation)
        props.area = props.top_width * props.depth;
        
        // Wetted perimeter (rectangular channel)
        props.wetted_perimeter = props.top_width + Real(2.0) * props.depth;
        
        // Hydraulic radius
        props.hydraulic_radius = props.area / props.wetted_perimeter;
        
        // Celerity factor for power-law geometry
        // c = (5/3 - 2*b/3) * V where b is width exponent
        // This accounts for how width changes with depth
        double b = to_double(power_law.width_exp);
        props.celerity_factor = Real(5.0/3.0 - 2.0*b/3.0);
        
        // Bound celerity factor to reasonable range [1.0, 5/3]
        if (to_double(props.celerity_factor) < 1.0) props.celerity_factor = Real(1.0);
        if (to_double(props.celerity_factor) > 5.0/3.0) props.celerity_factor = Real(5.0/3.0);
        
        return props;
    }
    
    /**
     * Trapezoidal channel geometry
     * 
     * A = (b + z*y) * y  where b=bottom width, z=side slope, y=depth
     * P = b + 2*y*sqrt(1 + z²)
     * R = A / P
     * T = b + 2*z*y  (top width)
     * 
     * For trapezoidal: c = (5/3) * V * [1 - (4/3) * R * z / (b + 2*z*y)]
     */
    HydraulicProps compute_trapezoidal(Real Q, double S, Real n) const {
        HydraulicProps props;
        
        Real Q_safe = safe_max(Q, Real(0.001));
        Real b = trapezoidal.bottom_width;
        Real z = trapezoidal.side_slope;
        
        // Solve for depth using Manning's equation iteratively
        // Q = (1/n) * A * R^(2/3) * S^(1/2)
        // Start with initial guess
        Real y = Real(1.0);  // Initial depth guess
        Real S_sqrt = Real(std::sqrt(S));
        
        // Newton-Raphson iteration (5 iterations usually sufficient)
        for (int iter = 0; iter < 8; ++iter) {
            Real A = (b + z * y) * y;
            Real P = b + Real(2.0) * y * Real(std::sqrt(1.0 + to_double(z*z)));
            Real R = A / P;
            
            // Q_calc from Manning
            Real Q_calc = (Real(1.0) / n) * A * pow(R, Real(2.0/3.0)) * S_sqrt;
            
            // dQ/dy for Newton step
            Real T = b + Real(2.0) * z * y;  // Top width
            Real dA_dy = T;
            Real dP_dy = Real(2.0) * Real(std::sqrt(1.0 + to_double(z*z)));
            Real dR_dy = (dA_dy * P - A * dP_dy) / (P * P);
            Real dQ_dy = (Real(1.0) / n) * S_sqrt * 
                         (dA_dy * pow(R, Real(2.0/3.0)) + 
                          A * Real(2.0/3.0) * pow(R, Real(-1.0/3.0)) * dR_dy);
            
            // Newton update
            Real dy = (Q_safe - Q_calc) / dQ_dy;
            y = y + dy;
            
            // Bound depth
            if (to_double(y) < 0.01) y = Real(0.01);
            if (to_double(y) > to_double(trapezoidal.max_depth)) 
                y = trapezoidal.max_depth;
            
            // Check convergence
            if (std::abs(to_double(dy)) < 0.001) break;
        }
        
        // Final properties
        props.depth = y;
        props.area = (b + z * y) * y;
        props.wetted_perimeter = b + Real(2.0) * y * Real(std::sqrt(1.0 + to_double(z*z)));
        props.hydraulic_radius = props.area / props.wetted_perimeter;
        props.top_width = b + Real(2.0) * z * y;
        
        // Celerity factor for trapezoidal channel
        // c = (5/3) * V * [1 - (4/3) * (A/P) * (dP/dA - 1/T)]
        // Simplified: c ≈ (5/3) * V * correction_factor
        Real P = props.wetted_perimeter;
        Real A = props.area;
        Real T = props.top_width;
        Real correction = Real(1.0) - Real(4.0/3.0) * (A / P) * 
                          (Real(2.0 * std::sqrt(1.0 + to_double(z*z))) / T - Real(1.0) / T);
        props.celerity_factor = Real(5.0/3.0) * correction;
        
        // Bound to reasonable range
        if (to_double(props.celerity_factor) < 1.0) props.celerity_factor = Real(1.0);
        if (to_double(props.celerity_factor) > 2.0) props.celerity_factor = Real(2.0);
        
        return props;
    }
    
    /**
     * Compound channel (main channel + floodplain)
     * 
     * Below bankfull: behaves as trapezoidal main channel
     * Above bankfull: adds rectangular floodplain sections
     * 
     * Uses divided channel method with separate conveyances
     */
    HydraulicProps compute_compound(Real Q, double S, Real n) const {
        HydraulicProps props;
        
        Real Q_safe = safe_max(Q, Real(0.001));
        
        // First, compute bankfull discharge
        Real y_bf = compound.bankfull_depth;
        Real b = compound.main_bottom_width;
        Real z = compound.main_side_slope;
        
        Real A_bf = (b + z * y_bf) * y_bf;
        Real P_bf = b + Real(2.0) * y_bf * Real(std::sqrt(1.0 + to_double(z*z)));
        Real R_bf = A_bf / P_bf;
        Real S_sqrt = Real(std::sqrt(S));
        Real Q_bf = (Real(1.0) / n) * A_bf * pow(R_bf, Real(2.0/3.0)) * S_sqrt;
        
        if (to_double(Q_safe) <= to_double(Q_bf)) {
            // Below bankfull - use trapezoidal solution
            TrapezoidalParams temp_trap;
            temp_trap.bottom_width = b;
            temp_trap.side_slope = z;
            temp_trap.max_depth = y_bf;
            
            ChannelGeometry temp_geom;
            temp_geom.type = ChannelType::TRAPEZOIDAL;
            temp_geom.trapezoidal = temp_trap;
            return temp_geom.compute_trapezoidal(Q_safe, S, n);
        }
        
        // Above bankfull - compound channel
        // Solve for total depth iteratively
        Real y = y_bf + Real(0.5);  // Start above bankfull
        Real W_fp = compound.floodplain_width;
        Real n_fp = n * compound.floodplain_roughness_factor;
        
        for (int iter = 0; iter < 8; ++iter) {
            // Main channel (full trapezoidal to bankfull + rectangular above)
            Real y_above = y - y_bf;
            Real T_main = b + Real(2.0) * z * y_bf;  // Top width at bankfull
            
            Real A_main = A_bf + T_main * y_above;
            Real P_main = P_bf;  // Wetted perimeter only at bankfull
            Real R_main = A_main / P_main;
            Real Q_main = (Real(1.0) / n) * A_main * pow(R_main, Real(2.0/3.0)) * S_sqrt;
            
            // Floodplain (rectangular, both sides)
            Real A_fp = W_fp * y_above;
            Real P_fp = W_fp + Real(2.0) * y_above;
            Real R_fp = A_fp / P_fp;
            Real Q_fp = (Real(1.0) / n_fp) * A_fp * pow(R_fp, Real(2.0/3.0)) * S_sqrt;
            
            Real Q_total = Q_main + Q_fp;
            
            // dQ/dy for Newton step
            Real dA_main_dy = T_main;
            Real dQ_main_dy = (Real(1.0) / n) * S_sqrt * dA_main_dy * pow(R_main, Real(2.0/3.0));
            
            Real dA_fp_dy = W_fp;
            Real dP_fp_dy = Real(2.0);
            Real dR_fp_dy = (dA_fp_dy * P_fp - A_fp * dP_fp_dy) / (P_fp * P_fp);
            Real dQ_fp_dy = (Real(1.0) / n_fp) * S_sqrt * 
                            (dA_fp_dy * pow(R_fp, Real(2.0/3.0)) +
                             A_fp * Real(2.0/3.0) * pow(R_fp, Real(-1.0/3.0)) * dR_fp_dy);
            
            Real dQ_dy = dQ_main_dy + dQ_fp_dy;
            Real dy = (Q_safe - Q_total) / dQ_dy;
            y = y + dy;
            
            if (to_double(y) < to_double(y_bf) + 0.01) y = y_bf + Real(0.01);
            if (std::abs(to_double(dy)) < 0.001) break;
        }
        
        // Final properties for compound channel
        Real y_above = y - y_bf;
        Real T_main = b + Real(2.0) * z * y_bf;
        
        props.depth = y;
        props.area = A_bf + T_main * y_above + W_fp * y_above;
        props.wetted_perimeter = P_bf + W_fp + Real(2.0) * y_above;
        props.hydraulic_radius = props.area / props.wetted_perimeter;
        props.top_width = T_main + W_fp;
        
        // Celerity factor for compound channel (approximation)
        // Use conveyance-weighted average
        props.celerity_factor = Real(5.0/3.0);  // Simplified
        
        return props;
    }
};

} // namespace dmc

#endif // DMC_CHANNEL_GEOMETRY_HPP
