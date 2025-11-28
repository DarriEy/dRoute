/**
 * @file runoff_forcing.hpp
 * @brief Generic runoff forcing reader with mizuRoute-style configuration.
 * 
 * Model-agnostic reader that can be configured for SUMMA, FUSE, GR4J, or
 * any other hydrological model output. Uses a control/config structure
 * similar to mizuRoute.
 */

#ifndef DMC_ROUTE_RUNOFF_FORCING_HPP
#define DMC_ROUTE_RUNOFF_FORCING_HPP

#include "types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <sstream>

#ifdef DMC_USE_NETCDF
    #ifdef DMC_USE_NETCDF_CXX4
        #include <netcdf>
    #else
        #include <netcdf.h>
    #endif
#endif

namespace dmc {

/**
 * Configuration for runoff forcing - mirrors mizuRoute control file variables.
 * 
 * These match mizuRoute naming conventions:
 * - vname_* = variable name
 * - dname_* = dimension name
 * - units_* = units specification
 */
struct RunoffForcingConfig {
    // === File paths (set via control file or command line) ===
    std::string network_file = "";        // <fname_ntopo>  topology.nc path
    std::string forcing_file = "";        // <fname_qsim>   runoff NetCDF path  
    std::string output_file = "";         // <fname_output> discharge output CSV
    std::string jacobian_file = "";       // <fname_jacobian> Jacobian output CSV
    
    // === Simulation options ===
    bool enable_gradients = false;        // <enable_gradients> AD gradient computation
    double dt = 3600.0;                   // <dt_routing> Routing timestep [seconds]
    std::string routing_method = "muskingum";  // <routing_method> muskingum, irf, kwt
    
    // === Runoff variable (the main input) ===
    std::string vname_qsim = "averageRoutedRunoff";  // Variable name for runoff
    std::string units_qsim = "m/s";                   // Units: "m/s", "m3/s", "mm/h", "mm/d"
    double dt_qsim = 3600.0;                          // Time interval of runoff [seconds]
    
    // === Time dimension/variable ===
    std::string dname_time = "time";
    std::string vname_time = "time";
    
    // === HRU/GRU dimension/variable ===
    std::string dname_hruid = "gru";      // Dimension name (could be "hru", "gru", "catchment")
    std::string vname_hruid = "gruId";    // Variable name for HRU/GRU IDs
    
    // === Optional: HRU area (needed if units are depth-based like m/s) ===
    std::string vname_area = "";          // If empty, use areas from topology
    
    // === Time subsetting ===
    size_t start_index = 0;
    size_t end_index = SIZE_MAX;          // SIZE_MAX = all timesteps
    
    // === Fallback variables (summed if main variable not found) ===
    std::vector<std::string> fallback_vars = {};  // e.g., {"scalarSurfaceRunoff", "scalarAquiferBaseflow"}
    
    /**
     * Load configuration from a control file (mizuRoute-style).
     * 
     * Format: <tag> value ! comment
     * Example:
     *   <vname_qsim>  averageRoutedRunoff  ! runoff variable name
     *   <units_qsim>  m/s                   ! runoff units
     */
    static RunoffForcingConfig load_from_control_file(const std::string& filepath);
    
    /**
     * Create preset for SUMMA output.
     */
    static RunoffForcingConfig summa_preset() {
        RunoffForcingConfig cfg;
        cfg.vname_qsim = "averageRoutedRunoff";
        cfg.units_qsim = "m/s";
        cfg.dname_hruid = "gru";
        cfg.vname_hruid = "gruId";
        return cfg;
    }
    
    /**
     * Create preset for FUSE output.
     */
    static RunoffForcingConfig fuse_preset() {
        RunoffForcingConfig cfg;
        cfg.vname_qsim = "q_routed";
        cfg.units_qsim = "mm/d";
        cfg.dname_hruid = "gru";
        cfg.vname_hruid = "gruId";
        return cfg;
    }
    
    /**
     * Create preset for GR4J output.
     */
    static RunoffForcingConfig gr4j_preset() {
        RunoffForcingConfig cfg;
        cfg.vname_qsim = "q_routed";
        cfg.units_qsim = "mm/d";
        cfg.dname_hruid = "gru";
        cfg.vname_hruid = "gruId";
        return cfg;
    }
};

/**
 * Lateral inflow timeseries - the output of forcing readers.
 */
struct LateralInflowData {
    std::vector<double> times;  // Time values
    std::unordered_map<int, std::vector<double>> reach_inflows;  // reach_id -> [m³/s] per timestep
    double dt = 3600.0;  // Timestep in seconds
    
    size_t num_timesteps() const { return times.size(); }
    
    double get_inflow(int reach_id, size_t timestep) const {
        auto it = reach_inflows.find(reach_id);
        if (it == reach_inflows.end() || timestep >= it->second.size()) {
            return 0.0;
        }
        return it->second[timestep];
    }
};

/**
 * HRU-to-reach mapping with areas.
 */
struct HRUMapping {
    std::unordered_map<int, int> hru_to_reach;      // hru_id -> reach_id
    std::unordered_map<int, double> hru_areas;      // hru_id -> area [m²]
    
    /**
     * Load from topology.nc (mizuRoute format).
     */
    static HRUMapping load_from_topology_nc(const std::string& filepath,
                                             const std::string& hru_id_var = "hruId",
                                             const std::string& hru_to_seg_var = "hruToSegId",
                                             const std::string& area_var = "area");
    
    /**
     * Load from CSV.
     * Format: hru_id,reach_id,area
     */
    static HRUMapping load_from_csv(const std::string& filepath);
    
    /**
     * Create identity mapping (hru_id == reach_id).
     */
    static HRUMapping create_identity(const std::vector<int>& ids, double default_area = 1.0);
};

/**
 * Generic runoff forcing reader.
 * 
 * Reads runoff from any hydrological model output and converts to
 * lateral inflows per reach. Configuration-driven, model-agnostic.
 */
class RunoffForcingReader {
public:
    explicit RunoffForcingReader(RunoffForcingConfig config = RunoffForcingConfig());
    
    /**
     * Set HRU mapping (required before read()).
     */
    void set_mapping(const HRUMapping& mapping) { mapping_ = mapping; mapping_set_ = true; }
    
    /**
     * Read forcing file and compute lateral inflows.
     */
    LateralInflowData read(const std::string& filepath);
    
    /**
     * Read single timestep (for online coupling).
     */
    std::unordered_map<int, double> read_timestep(const std::string& filepath, size_t timestep_idx);

private:
    RunoffForcingConfig config_;
    HRUMapping mapping_;
    bool mapping_set_ = false;
    
    double convert_to_m3_per_s(double value, double area_m2) const;
    
    std::unordered_map<int, double> aggregate_to_reaches(
        const std::vector<int>& hru_ids,
        const std::vector<double>& runoff_values,
        const std::vector<double>& areas);
};

// ==================== Implementation ====================

inline RunoffForcingConfig RunoffForcingConfig::load_from_control_file(const std::string& filepath) {
    RunoffForcingConfig cfg;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open control file: " + filepath);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '!') continue;
        
        // Parse <tag> value format
        size_t tag_start = line.find('<');
        size_t tag_end = line.find('>');
        if (tag_start == std::string::npos || tag_end == std::string::npos) continue;
        
        std::string tag = line.substr(tag_start + 1, tag_end - tag_start - 1);
        
        // Extract value (everything after '>' until '!' or end)
        std::string rest = line.substr(tag_end + 1);
        size_t comment_pos = rest.find('!');
        if (comment_pos != std::string::npos) {
            rest = rest.substr(0, comment_pos);
        }
        
        // Trim whitespace
        size_t start = rest.find_first_not_of(" \t");
        size_t end = rest.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        std::string value = rest.substr(start, end - start + 1);
        
        // Set config values - file paths
        if (tag == "fname_ntopo" || tag == "network_file") cfg.network_file = value;
        else if (tag == "fname_qsim" || tag == "forcing_file") cfg.forcing_file = value;
        else if (tag == "fname_output" || tag == "output_file") cfg.output_file = value;
        else if (tag == "fname_jacobian" || tag == "jacobian_file") cfg.jacobian_file = value;
        
        // Simulation options
        else if (tag == "enable_gradients") cfg.enable_gradients = (value == "true" || value == "yes" || value == "1");
        else if (tag == "dt_routing" || tag == "dt") cfg.dt = std::stod(value);
        else if (tag == "routing_method") cfg.routing_method = value;
        
        // Runoff variable config
        else if (tag == "vname_qsim") cfg.vname_qsim = value;
        else if (tag == "units_qsim") cfg.units_qsim = value;
        else if (tag == "dt_qsim") cfg.dt_qsim = std::stod(value);
        else if (tag == "dname_time") cfg.dname_time = value;
        else if (tag == "vname_time") cfg.vname_time = value;
        else if (tag == "dname_hruid") cfg.dname_hruid = value;
        else if (tag == "vname_hruid") cfg.vname_hruid = value;
        else if (tag == "vname_area") cfg.vname_area = value;
    }
    
    return cfg;
}

inline HRUMapping HRUMapping::load_from_csv(const std::string& filepath) {
    HRUMapping mapping;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open mapping file: " + filepath);
    }
    
    std::string line;
    bool header = true;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Skip header
        if (header) {
            header = false;
            if (line.find("hru") != std::string::npos || line.find("HRU") != std::string::npos) {
                continue;
            }
        }
        
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 2) {
            int hru_id = std::stoi(tokens[0]);
            int reach_id = std::stoi(tokens[1]);
            double area = (tokens.size() >= 3) ? std::stod(tokens[2]) : 1.0;
            
            mapping.hru_to_reach[hru_id] = reach_id;
            mapping.hru_areas[hru_id] = area;
        }
    }
    
    return mapping;
}

inline HRUMapping HRUMapping::create_identity(const std::vector<int>& ids, double default_area) {
    HRUMapping mapping;
    for (int id : ids) {
        mapping.hru_to_reach[id] = id;
        mapping.hru_areas[id] = default_area;
    }
    return mapping;
}

#ifdef DMC_USE_NETCDF
inline HRUMapping HRUMapping::load_from_topology_nc(
    const std::string& filepath,
    const std::string& hru_id_var,
    const std::string& hru_to_seg_var,
    const std::string& area_var) {
    
    HRUMapping mapping;
    
#ifdef DMC_USE_NETCDF_CXX4
    netCDF::NcFile nc(filepath, netCDF::NcFile::read);
    
    // Find HRU dimension
    auto dims = nc.getDims();
    size_t num_hru = 0;
    for (const auto& [name, dim] : dims) {
        if (name == "hru" || name == "seg") {
            num_hru = dim.getSize();
            break;
        }
    }
    
    std::vector<int> hru_ids(num_hru);
    std::vector<int> hru_to_seg(num_hru);
    std::vector<double> areas(num_hru);
    
    auto id_var = nc.getVar(hru_id_var);
    if (id_var.getType().getSize() == 8) {
        std::vector<long long> temp(num_hru);
        id_var.getVar(temp.data());
        for (size_t i = 0; i < num_hru; ++i) hru_ids[i] = static_cast<int>(temp[i]);
    } else {
        id_var.getVar(hru_ids.data());
    }
    
    auto seg_var = nc.getVar(hru_to_seg_var);
    if (seg_var.getType().getSize() == 8) {
        std::vector<long long> temp(num_hru);
        seg_var.getVar(temp.data());
        for (size_t i = 0; i < num_hru; ++i) hru_to_seg[i] = static_cast<int>(temp[i]);
    } else {
        seg_var.getVar(hru_to_seg.data());
    }
    
    nc.getVar(area_var).getVar(areas.data());
    
#else
    int ncid;
    nc_open(filepath.c_str(), NC_NOWRITE, &ncid);
    
    // Get HRU dimension size
    int hru_dimid;
    size_t num_hru;
    if (nc_inq_dimid(ncid, "hru", &hru_dimid) != NC_NOERR) {
        nc_inq_dimid(ncid, "seg", &hru_dimid);  // Fallback
    }
    nc_inq_dimlen(ncid, hru_dimid, &num_hru);
    
    std::vector<int> hru_ids(num_hru);
    std::vector<int> hru_to_seg(num_hru);
    std::vector<double> areas(num_hru);
    
    // Read variables
    int varid;
    nc_type vartype;
    
    nc_inq_varid(ncid, hru_id_var.c_str(), &varid);
    nc_inq_vartype(ncid, varid, &vartype);
    if (vartype == NC_INT64) {
        std::vector<long long> temp(num_hru);
        nc_get_var_longlong(ncid, varid, temp.data());
        for (size_t i = 0; i < num_hru; ++i) hru_ids[i] = static_cast<int>(temp[i]);
    } else {
        nc_get_var_int(ncid, varid, hru_ids.data());
    }
    
    nc_inq_varid(ncid, hru_to_seg_var.c_str(), &varid);
    nc_inq_vartype(ncid, varid, &vartype);
    if (vartype == NC_INT64) {
        std::vector<long long> temp(num_hru);
        nc_get_var_longlong(ncid, varid, temp.data());
        for (size_t i = 0; i < num_hru; ++i) hru_to_seg[i] = static_cast<int>(temp[i]);
    } else {
        nc_get_var_int(ncid, varid, hru_to_seg.data());
    }
    
    nc_inq_varid(ncid, area_var.c_str(), &varid);
    nc_get_var_double(ncid, varid, areas.data());
    
    nc_close(ncid);
#endif
    
    for (size_t i = 0; i < num_hru; ++i) {
        mapping.hru_to_reach[hru_ids[i]] = hru_to_seg[i];
        mapping.hru_areas[hru_ids[i]] = areas[i];
    }
    
    return mapping;
}
#else
inline HRUMapping HRUMapping::load_from_topology_nc(
    const std::string&, const std::string&, const std::string&, const std::string&) {
    throw std::runtime_error("NetCDF support not compiled");
}
#endif

inline RunoffForcingReader::RunoffForcingReader(RunoffForcingConfig config)
    : config_(std::move(config)) {}

inline double RunoffForcingReader::convert_to_m3_per_s(double value, double area_m2) const {
    if (config_.units_qsim == "m3/s" || config_.units_qsim == "m^3/s") {
        return value;  // Already in m³/s
    } else if (config_.units_qsim == "m/s") {
        return value * area_m2;  // [m/s] * [m²] = [m³/s]
    } else if (config_.units_qsim == "mm/s") {
        return value * 0.001 * area_m2;  // [mm/s] -> [m/s] -> [m³/s]
    } else if (config_.units_qsim == "mm/h") {
        return value * 0.001 / 3600.0 * area_m2;
    } else if (config_.units_qsim == "mm/d") {
        return value * 0.001 / 86400.0 * area_m2;
    } else {
        // Assume m/s as default
        return value * area_m2;
    }
}

inline std::unordered_map<int, double> RunoffForcingReader::aggregate_to_reaches(
    const std::vector<int>& hru_ids,
    const std::vector<double>& runoff_values,
    const std::vector<double>& areas) {
    
    std::unordered_map<int, double> reach_inflows;
    
    for (size_t i = 0; i < hru_ids.size(); ++i) {
        int hru_id = hru_ids[i];
        double runoff = runoff_values[i];
        double area = areas[i];
        
        // Find which reach this HRU drains to
        auto it = mapping_.hru_to_reach.find(hru_id);
        if (it == mapping_.hru_to_reach.end()) continue;
        
        int reach_id = it->second;
        
        // Convert to m³/s and add
        double flow = convert_to_m3_per_s(runoff, area);
        reach_inflows[reach_id] += flow;
    }
    
    return reach_inflows;
}

inline LateralInflowData RunoffForcingReader::read(const std::string& filepath) {
#ifndef DMC_USE_NETCDF
    throw std::runtime_error("NetCDF support not compiled");
#else
    if (!mapping_set_) {
        throw std::runtime_error("HRU mapping not set. Call set_mapping() first.");
    }
    
    LateralInflowData result;
    result.dt = config_.dt_qsim;
    
#ifdef DMC_USE_NETCDF_CXX4
    netCDF::NcFile nc(filepath, netCDF::NcFile::read);
    
    size_t num_time = nc.getDim(config_.dname_time).getSize();
    size_t num_hru = nc.getDim(config_.dname_hruid).getSize();
    
#else
    int ncid;
    int retval = nc_open(filepath.c_str(), NC_NOWRITE, &ncid);
    if (retval != NC_NOERR) {
        throw std::runtime_error("Failed to open " + filepath + ": " + nc_strerror(retval));
    }
    
    int time_dimid, hru_dimid;
    size_t num_time, num_hru;
    
    nc_inq_dimid(ncid, config_.dname_time.c_str(), &time_dimid);
    nc_inq_dimlen(ncid, time_dimid, &num_time);
    
    nc_inq_dimid(ncid, config_.dname_hruid.c_str(), &hru_dimid);
    nc_inq_dimlen(ncid, hru_dimid, &num_hru);
#endif
    
    // Apply time subsetting
    size_t start_t = config_.start_index;
    size_t end_t = std::min(config_.end_index, num_time);
    size_t num_steps = end_t - start_t;
    
    if (num_steps == 0) {
        // No data
#ifndef DMC_USE_NETCDF_CXX4
        nc_close(ncid);
#endif
        return result;
    }
    
    // Read HRU IDs
    std::vector<int> hru_ids(num_hru);
    
#ifdef DMC_USE_NETCDF_CXX4
    auto hru_id_var = nc.getVar(config_.vname_hruid);
    if (hru_id_var.getType().getSize() == 8) {
        std::vector<long long> temp(num_hru);
        hru_id_var.getVar(temp.data());
        for (size_t i = 0; i < num_hru; ++i) hru_ids[i] = static_cast<int>(temp[i]);
    } else {
        hru_id_var.getVar(hru_ids.data());
    }
#else
    int hru_id_varid;
    nc_inq_varid(ncid, config_.vname_hruid.c_str(), &hru_id_varid);
    nc_type vartype;
    nc_inq_vartype(ncid, hru_id_varid, &vartype);
    if (vartype == NC_INT64) {
        std::vector<long long> temp(num_hru);
        nc_get_var_longlong(ncid, hru_id_varid, temp.data());
        for (size_t i = 0; i < num_hru; ++i) hru_ids[i] = static_cast<int>(temp[i]);
    } else {
        nc_get_var_int(ncid, hru_id_varid, hru_ids.data());
    }
#endif
    
    // Get areas (from mapping or from file)
    std::vector<double> areas(num_hru);
    for (size_t i = 0; i < num_hru; ++i) {
        auto it = mapping_.hru_areas.find(hru_ids[i]);
        areas[i] = (it != mapping_.hru_areas.end()) ? it->second : 1.0;
    }
    
    // Read runoff data
    std::vector<double> runoff_data(num_steps * num_hru);
    
#ifdef DMC_USE_NETCDF_CXX4
    auto runoff_var = nc.getVar(config_.vname_qsim);
    std::vector<size_t> start = {start_t, 0};
    std::vector<size_t> count = {num_steps, num_hru};
    runoff_var.getVar(start, count, runoff_data.data());
    
    // Read time
    result.times.resize(num_steps);
    auto time_var = nc.getVar(config_.vname_time);
    std::vector<size_t> time_start = {start_t};
    std::vector<size_t> time_count = {num_steps};
    time_var.getVar(time_start, time_count, result.times.data());
#else
    int runoff_varid;
    nc_inq_varid(ncid, config_.vname_qsim.c_str(), &runoff_varid);
    size_t start[2] = {start_t, 0};
    size_t count[2] = {num_steps, num_hru};
    nc_get_vara_double(ncid, runoff_varid, start, count, runoff_data.data());
    
    // Read time
    result.times.resize(num_steps);
    int time_varid;
    nc_inq_varid(ncid, config_.vname_time.c_str(), &time_varid);
    size_t time_start[1] = {start_t};
    size_t time_count[1] = {num_steps};
    nc_get_vara_double(ncid, time_varid, time_start, time_count, result.times.data());
    
    nc_close(ncid);
#endif
    
    // Process each timestep
    for (size_t t = 0; t < num_steps; ++t) {
        std::vector<double> runoff_t(num_hru);
        for (size_t h = 0; h < num_hru; ++h) {
            runoff_t[h] = runoff_data[t * num_hru + h];
        }
        
        auto reach_flows = aggregate_to_reaches(hru_ids, runoff_t, areas);
        
        for (const auto& [reach_id, flow] : reach_flows) {
            result.reach_inflows[reach_id].push_back(flow);
        }
    }
    
    // Ensure all reaches have same number of timesteps
    for (auto& [reach_id, inflows] : result.reach_inflows) {
        while (inflows.size() < num_steps) {
            inflows.push_back(0.0);
        }
    }
    
    return result;
#endif
}

inline std::unordered_map<int, double> RunoffForcingReader::read_timestep(
    const std::string& filepath, size_t timestep_idx) {
    
    // For single timestep, just read full and extract
    // (Could optimize later for online coupling)
    auto full = read(filepath);
    
    std::unordered_map<int, double> result;
    for (const auto& [reach_id, inflows] : full.reach_inflows) {
        if (timestep_idx < inflows.size()) {
            result[reach_id] = inflows[timestep_idx];
        }
    }
    
    return result;
}

} // namespace dmc

#endif // DMC_ROUTE_RUNOFF_FORCING_HPP
