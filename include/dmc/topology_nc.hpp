/**
 * @file topology_nc.hpp
 * @brief Reader for mizuRoute-style topology.nc files.
 * 
 * This allows dMC-Route to use the same topology files as mizuRoute,
 * enabling drop-in replacement in SYMFLUENCE workflows.
 */

#ifndef DMC_ROUTE_TOPOLOGY_NC_HPP
#define DMC_ROUTE_TOPOLOGY_NC_HPP

#include "network.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>

#ifdef DMC_USE_NETCDF
    #ifdef DMC_USE_NETCDF_CXX4
        #include <netcdf>
    #else
        #include <netcdf.h>
    #endif
#endif

namespace dmc {

/**
 * Configuration for reading topology.nc files.
 */
struct TopologyConfig {
    // Variable names in the NetCDF file (mizuRoute defaults)
    std::string seg_id_var = "segId";
    std::string down_seg_id_var = "downSegId";
    std::string slope_var = "slope";
    std::string length_var = "length";
    std::string hru_id_var = "hruId";
    std::string hru_to_seg_var = "hruToSegId";
    std::string area_var = "area";
    
    // Dimension names
    std::string seg_dim = "seg";
    std::string hru_dim = "hru";
    
    // Default Manning's n (not in topology.nc, must be set)
    double default_manning_n = 0.035;
    
    // Default channel geometry parameters (Leopold & Maddock)
    double default_width_coef = 7.2;
    double default_width_exp = 0.5;
    double default_depth_coef = 0.27;
    double default_depth_exp = 0.3;
    
    // Minimum values to prevent numerical issues
    double min_slope = 1e-6;
    double min_length = 1.0;  // meters
};

/**
 * HRU (Hydrologic Response Unit) information for forcing aggregation.
 */
struct HRUInfo {
    int hru_id;           // HRU identifier
    int segment_id;       // Which segment this HRU drains to
    double area_m2;       // Area in square meters
};

/**
 * Reader for mizuRoute-style topology.nc files.
 * 
 * Creates a dMC Network from topology.nc, which contains:
 * - segId: segment identifiers
 * - downSegId: downstream segment ID (0 = outlet)
 * - slope: segment slopes [m/m]
 * - length: segment lengths [m]
 * - hruId: HRU identifiers
 * - hruToSegId: mapping from HRU to segment
 * - area: HRU areas [m²]
 */
class TopologyNCReader {
public:
    explicit TopologyNCReader(TopologyConfig config = TopologyConfig());
    
    /**
     * Load network from topology.nc file.
     * 
     * @param filepath Path to topology.nc
     * @return Network object ready for routing
     */
    Network load_network(const std::string& filepath);
    
    /**
     * Get HRU information for forcing aggregation.
     * Call after load_network().
     * 
     * @return Vector of HRU info (id, segment, area)
     */
    const std::vector<HRUInfo>& get_hru_info() const { return hru_info_; }
    
    /**
     * Get mapping from HRU ID to segment ID.
     * Useful for mapping SUMMA GRU output to reaches.
     */
    const std::unordered_map<int, int>& get_hru_to_segment_map() const { 
        return hru_to_segment_; 
    }
    
    /**
     * Get HRU areas indexed by HRU ID.
     */
    const std::unordered_map<int, double>& get_hru_areas() const {
        return hru_areas_;
    }
    
    /**
     * Get Manning's n values indexed by segment ID.
     * Returns empty if not read from file (defaults used).
     */
    const std::unordered_map<int, double>& get_manning_n() const {
        return manning_n_values_;
    }

private:
    TopologyConfig config_;
    std::vector<HRUInfo> hru_info_;
    std::unordered_map<int, int> hru_to_segment_;
    std::unordered_map<int, double> hru_areas_;
    std::unordered_map<int, double> manning_n_values_;  // Optional: from topology.nc
    
#ifdef DMC_USE_NETCDF
    void read_netcdf(const std::string& filepath,
                     std::vector<int>& seg_ids,
                     std::vector<int>& down_seg_ids,
                     std::vector<double>& slopes,
                     std::vector<double>& lengths,
                     std::vector<int>& hru_ids,
                     std::vector<int>& hru_to_seg,
                     std::vector<double>& areas);
#endif
    
    Network build_network(const std::vector<int>& seg_ids,
                          const std::vector<int>& down_seg_ids,
                          const std::vector<double>& slopes,
                          const std::vector<double>& lengths);
};

// ==================== Implementation ====================

inline TopologyNCReader::TopologyNCReader(TopologyConfig config)
    : config_(std::move(config)) {}

inline Network TopologyNCReader::load_network(const std::string& filepath) {
#ifndef DMC_USE_NETCDF
    throw std::runtime_error("NetCDF support not compiled. Rebuild with -DDMC_ENABLE_NETCDF=ON");
#else
    std::vector<int> seg_ids, down_seg_ids, hru_ids, hru_to_seg;
    std::vector<double> slopes, lengths, areas;
    
    read_netcdf(filepath, seg_ids, down_seg_ids, slopes, lengths,
                hru_ids, hru_to_seg, areas);
    
    // Try to read optional mann_n variable
    manning_n_values_.clear();
    try {
        int ncid;
        if (nc_open(filepath.c_str(), NC_NOWRITE, &ncid) == NC_NOERR) {
            int varid;
            if (nc_inq_varid(ncid, "mann_n", &varid) == NC_NOERR) {
                std::vector<double> mann_n(seg_ids.size());
                nc_get_var_double(ncid, varid, mann_n.data());
                for (size_t i = 0; i < seg_ids.size(); ++i) {
                    manning_n_values_[seg_ids[i]] = mann_n[i];
                }
            }
            nc_close(ncid);
        }
    } catch (...) {
        // Ignore errors - use defaults
    }
    
    // Store HRU information for forcing mapping
    hru_info_.clear();
    hru_to_segment_.clear();
    hru_areas_.clear();
    
    for (size_t i = 0; i < hru_ids.size(); ++i) {
        HRUInfo info;
        info.hru_id = hru_ids[i];
        info.segment_id = hru_to_seg[i];
        info.area_m2 = areas[i];
        hru_info_.push_back(info);
        
        hru_to_segment_[hru_ids[i]] = hru_to_seg[i];
        hru_areas_[hru_ids[i]] = areas[i];
    }
    
    return build_network(seg_ids, down_seg_ids, slopes, lengths);
#endif
}

#ifdef DMC_USE_NETCDF
inline void TopologyNCReader::read_netcdf(
    const std::string& filepath,
    std::vector<int>& seg_ids,
    std::vector<int>& down_seg_ids,
    std::vector<double>& slopes,
    std::vector<double>& lengths,
    std::vector<int>& hru_ids,
    std::vector<int>& hru_to_seg,
    std::vector<double>& areas) {
    
#ifdef DMC_USE_NETCDF_CXX4
    // netCDF-C++ 4 API
    netCDF::NcFile nc(filepath, netCDF::NcFile::read);
    
    // Get dimensions
    size_t num_seg = nc.getDim(config_.seg_dim).getSize();
    size_t num_hru = nc.getDim(config_.hru_dim).getSize();
    
    // Resize vectors
    seg_ids.resize(num_seg);
    down_seg_ids.resize(num_seg);
    slopes.resize(num_seg);
    lengths.resize(num_seg);
    hru_ids.resize(num_hru);
    hru_to_seg.resize(num_hru);
    areas.resize(num_hru);
    
    // Read segment variables
    nc.getVar(config_.seg_id_var).getVar(seg_ids.data());
    nc.getVar(config_.down_seg_id_var).getVar(down_seg_ids.data());
    nc.getVar(config_.slope_var).getVar(slopes.data());
    nc.getVar(config_.length_var).getVar(lengths.data());
    
    // Read HRU variables
    nc.getVar(config_.hru_id_var).getVar(hru_ids.data());
    nc.getVar(config_.hru_to_seg_var).getVar(hru_to_seg.data());
    nc.getVar(config_.area_var).getVar(areas.data());
    
#else
    // NetCDF C API
    int ncid;
    int retval = nc_open(filepath.c_str(), NC_NOWRITE, &ncid);
    if (retval != NC_NOERR) {
        throw std::runtime_error("Failed to open " + filepath + ": " + nc_strerror(retval));
    }
    
    // Get dimension sizes
    int seg_dimid, hru_dimid;
    size_t num_seg, num_hru;
    
    retval = nc_inq_dimid(ncid, config_.seg_dim.c_str(), &seg_dimid);
    if (retval != NC_NOERR) {
        nc_close(ncid);
        throw std::runtime_error("Failed to find dimension '" + config_.seg_dim + "': " + nc_strerror(retval));
    }
    nc_inq_dimlen(ncid, seg_dimid, &num_seg);
    
    retval = nc_inq_dimid(ncid, config_.hru_dim.c_str(), &hru_dimid);
    if (retval != NC_NOERR) {
        nc_close(ncid);
        throw std::runtime_error("Failed to find dimension '" + config_.hru_dim + "': " + nc_strerror(retval));
    }
    nc_inq_dimlen(ncid, hru_dimid, &num_hru);
    
    // Resize vectors
    seg_ids.resize(num_seg);
    down_seg_ids.resize(num_seg);
    slopes.resize(num_seg);
    lengths.resize(num_seg);
    hru_ids.resize(num_hru);
    hru_to_seg.resize(num_hru);
    areas.resize(num_hru);
    
    // Helper to read int variable with error checking
    auto read_int_var = [&](const std::string& name, std::vector<int>& data) {
        int varid;
        int ret = nc_inq_varid(ncid, name.c_str(), &varid);
        if (ret != NC_NOERR) {
            throw std::runtime_error("Failed to find variable '" + name + "': " + nc_strerror(ret));
        }
        
        // Check variable type - might be int64
        nc_type vartype;
        nc_inq_vartype(ncid, varid, &vartype);
        
        if (vartype == NC_INT64) {
            std::vector<long long> temp(data.size());
            nc_get_var_longlong(ncid, varid, temp.data());
            for (size_t i = 0; i < data.size(); ++i) {
                data[i] = static_cast<int>(temp[i]);
            }
        } else {
            nc_get_var_int(ncid, varid, data.data());
        }
    };
    
    // Helper to read double variable
    auto read_double_var = [&](const std::string& name, std::vector<double>& data) {
        int varid;
        nc_inq_varid(ncid, name.c_str(), &varid);
        nc_get_var_double(ncid, varid, data.data());
    };
    
    // Read segment variables
    read_int_var(config_.seg_id_var, seg_ids);
    read_int_var(config_.down_seg_id_var, down_seg_ids);
    read_double_var(config_.slope_var, slopes);
    read_double_var(config_.length_var, lengths);
    
    // Read HRU variables
    read_int_var(config_.hru_id_var, hru_ids);
    read_int_var(config_.hru_to_seg_var, hru_to_seg);
    read_double_var(config_.area_var, areas);
    
    nc_close(ncid);
#endif
}
#endif

inline Network TopologyNCReader::build_network(
    const std::vector<int>& seg_ids,
    const std::vector<int>& down_seg_ids,
    const std::vector<double>& slopes,
    const std::vector<double>& lengths) {
    
    Network net;
    
    // Build segment ID to index mapping
    std::unordered_map<int, size_t> seg_id_to_idx;
    for (size_t i = 0; i < seg_ids.size(); ++i) {
        seg_id_to_idx[seg_ids[i]] = i;
    }
    
    // Create reaches
    std::set<int> valid_seg_ids;
    for (size_t i = 0; i < seg_ids.size(); ++i) {
        valid_seg_ids.insert(seg_ids[i]);
        
        Reach r;
        r.id = seg_ids[i];
        r.name = "seg_" + std::to_string(seg_ids[i]);
        r.length = std::max(lengths[i], config_.min_length);
        r.slope = std::max(slopes[i], config_.min_slope);
        
        // Use Manning's n from file if available, otherwise use default
        auto it = manning_n_values_.find(seg_ids[i]);
        if (it != manning_n_values_.end()) {
            r.manning_n = Real(it->second);
        } else {
            r.manning_n = Real(config_.default_manning_n);
        }
        
        // Set default channel geometry
        r.geometry.width_coef = Real(config_.default_width_coef);
        r.geometry.width_exp = Real(config_.default_width_exp);
        r.geometry.depth_coef = Real(config_.default_depth_coef);
        r.geometry.depth_exp = Real(config_.default_depth_exp);
        
        // Downstream junction ID will be set after we know all valid segments
        r.downstream_junction_id = down_seg_ids[i];
        
        net.add_reach(r);
    }
    
    // Now fix downstream_junction_id for outlets
    // An outlet is when downSegId == 0 OR downSegId doesn't exist in valid_seg_ids
    for (size_t i = 0; i < seg_ids.size(); ++i) {
        int down_seg = down_seg_ids[i];
        if (down_seg == 0 || valid_seg_ids.find(down_seg) == valid_seg_ids.end()) {
            Reach& r = net.get_reach(seg_ids[i]);
            r.downstream_junction_id = -1;  // Mark as outlet
        }
    }
    
    // Create junctions based on connectivity
    // Each segment that receives flow needs a junction at its upstream end
    std::unordered_map<int, Junction> junctions;
    std::set<int> has_downstream_set;  // Track which downstream_reach_ids we've added
    
    for (size_t i = 0; i < seg_ids.size(); ++i) {
        int seg_id = seg_ids[i];
        int down_seg = down_seg_ids[i];
        
        // This segment flows into the junction at down_seg
        // Only if down_seg is a valid segment in our network
        if (down_seg != 0 && valid_seg_ids.count(down_seg)) {
            // Create or update junction at downstream segment
            if (junctions.find(down_seg) == junctions.end()) {
                Junction j;
                j.id = down_seg;
                j.name = "junc_" + std::to_string(down_seg);
                junctions[down_seg] = j;
            }
            junctions[down_seg].upstream_reach_ids.push_back(seg_id);
            
            // The downstream segment flows out of this junction (add only once)
            if (!has_downstream_set.count(down_seg)) {
                junctions[down_seg].downstream_reach_ids.push_back(down_seg);
                has_downstream_set.insert(down_seg);
            }
        }
    }
    
    // Identify headwaters (segments not receiving flow from upstream)
    std::set<int> receiving_segments;
    for (const auto& [junc_id, junc] : junctions) {
        for (int ds : junc.downstream_reach_ids) {
            receiving_segments.insert(ds);
        }
    }
    
    // Create headwater junctions for segments with no upstream
    for (int seg_id : valid_seg_ids) {
        if (receiving_segments.find(seg_id) == receiving_segments.end()) {
            // This segment is a headwater - create junction
            Junction j;
            j.id = -seg_id;  // Negative ID for headwater junctions
            j.name = "headwater_" + std::to_string(seg_id);
            j.is_headwater = true;
            j.downstream_reach_ids.push_back(seg_id);
            junctions[-seg_id] = j;
        }
    }
    
    // Mark outlet junctions - segments where downstream is outside network
    for (size_t i = 0; i < seg_ids.size(); ++i) {
        int seg_id = seg_ids[i];
        int down_seg = down_seg_ids[i];
        
        // Outlet if downSegId == 0 OR downSegId not in valid_seg_ids
        if (down_seg == 0 || valid_seg_ids.find(down_seg) == valid_seg_ids.end()) {
            // Find which junction this segment flows FROM
            for (auto& [junc_id, junc] : junctions) {
                auto it = std::find(junc.downstream_reach_ids.begin(), 
                                   junc.downstream_reach_ids.end(), 
                                   seg_id);
                if (it != junc.downstream_reach_ids.end()) {
                    // This junction feeds the outlet segment
                    break;
                }
            }
            
            // Also check if this is a headwater that's also an outlet
            int hw_junc_id = -seg_id;
            if (junctions.count(hw_junc_id)) {
                junctions[hw_junc_id].is_outlet = true;
            }
        }
    }
    
    // Set upstream junction IDs on reaches
    for (auto& [junc_id, junc] : junctions) {
        for (int ds_reach : junc.downstream_reach_ids) {
            if (valid_seg_ids.count(ds_reach)) {
                Reach& r = net.get_reach(ds_reach);
                r.upstream_junction_id = junc_id;
            }
        }
    }
    
    // Add junctions to network
    for (auto& [junc_id, junc] : junctions) {
        net.add_junction(junc);
    }
    
    // Build topology
    net.build_topology();
    
    return net;
}

} // namespace dmc

#endif // DMC_ROUTE_TOPOLOGY_NC_HPP
