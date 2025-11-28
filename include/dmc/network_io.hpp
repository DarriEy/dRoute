#ifndef DMC_ROUTE_NETWORK_IO_HPP
#define DMC_ROUTE_NETWORK_IO_HPP

#include "network.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <queue>
#include <set>

namespace dmc {

using json = nlohmann::json;

/**
 * Configuration for network loading.
 */
struct NetworkLoadConfig {
    // GeoJSON property names
    std::string reach_id_prop = "COMID";         // NHDPlus: COMID
    std::string reach_length_prop = "LENGTHKM";  // Length in km
    std::string reach_slope_prop = "SLOPE";      // Slope [m/m]
    std::string from_node_prop = "FromNode";     // Upstream node ID
    std::string to_node_prop = "ToNode";         // Downstream node ID
    std::string manning_n_prop = "";             // Optional: per-reach Manning's n
    
    // Unit conversions
    double length_to_meters = 1000.0;  // km to m
    double slope_factor = 1.0;
    
    // Default values
    double default_slope = 0.001;
    double default_manning_n = 0.035;
    double default_length = 1000.0;
    
    // Minimum slope to prevent numerical issues
    double min_slope = 1e-6;
};

/**
 * Load river network from various formats.
 */
class NetworkIO {
public:
    explicit NetworkIO(NetworkLoadConfig config = {}) : config_(std::move(config)) {}
    
    /**
     * Load network from GeoJSON file.
     * 
     * Expected format: FeatureCollection with LineString features representing reaches.
     * Each feature should have properties for reach ID, length, slope, and connectivity.
     * 
     * Example:
     * {
     *   "type": "FeatureCollection",
     *   "features": [
     *     {
     *       "type": "Feature",
     *       "geometry": {"type": "LineString", "coordinates": [[...], [...]]},
     *       "properties": {
     *         "COMID": 12345,
     *         "LENGTHKM": 5.2,
     *         "SLOPE": 0.001,
     *         "FromNode": 100,
     *         "ToNode": 101
     *       }
     *     },
     *     ...
     *   ]
     * }
     */
    Network load_geojson(const std::string& filepath);
    
    /**
     * Load network from NHDPlus-style flowline GeoJSON with separate attributes.
     */
    Network load_nhdplus(const std::string& flowlines_geojson,
                         const std::string& vaa_csv = "");
    
    /**
     * Load network from CSV files.
     * 
     * Reaches CSV format:
     *   reach_id, length_m, slope, from_node, to_node, manning_n
     * 
     * Optional parameters CSV:
     *   reach_id, manning_n, width_coef, width_exp, depth_coef, depth_exp
     */
    Network load_csv(const std::string& reaches_csv,
                     const std::string& params_csv = "");
    
    /**
     * Save network to GeoJSON (for visualization/debugging).
     */
    void save_geojson(const Network& network, const std::string& filepath);
    
    /**
     * Save network to CSV.
     */
    void save_csv(const Network& network, const std::string& filepath);
    
private:
    NetworkLoadConfig config_;
    
    void build_junctions(Network& network,
                         const std::unordered_map<int, int>& reach_from_node,
                         const std::unordered_map<int, int>& reach_to_node);
};

// ==================== Implementation ====================

inline Network NetworkIO::load_geojson(const std::string& filepath) {
    Network network;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open GeoJSON file: " + filepath);
    }
    
    json geojson;
    file >> geojson;
    
    if (geojson["type"] != "FeatureCollection") {
        throw std::runtime_error("Expected GeoJSON FeatureCollection");
    }
    
    // Maps for building topology
    std::unordered_map<int, int> reach_from_node;  // reach_id -> from_node_id
    std::unordered_map<int, int> reach_to_node;    // reach_id -> to_node_id
    
    for (const auto& feature : geojson["features"]) {
        if (feature["geometry"]["type"] != "LineString") {
            continue;  // Skip non-LineString features
        }
        
        const auto& props = feature["properties"];
        
        Reach reach;
        
        // Read reach ID
        if (props.contains(config_.reach_id_prop)) {
            reach.id = props[config_.reach_id_prop].get<int>();
        } else {
            throw std::runtime_error("Missing reach ID property: " + config_.reach_id_prop);
        }
        
        reach.name = "reach_" + std::to_string(reach.id);
        
        // Read length
        if (props.contains(config_.reach_length_prop)) {
            reach.length = props[config_.reach_length_prop].get<double>() * config_.length_to_meters;
        } else {
            reach.length = config_.default_length;
        }
        
        // Read slope
        if (props.contains(config_.reach_slope_prop)) {
            reach.slope = props[config_.reach_slope_prop].get<double>() * config_.slope_factor;
            reach.slope = std::max(reach.slope, config_.min_slope);
        } else {
            reach.slope = config_.default_slope;
        }
        
        // Read Manning's n if available
        if (!config_.manning_n_prop.empty() && props.contains(config_.manning_n_prop)) {
            reach.manning_n = Real(props[config_.manning_n_prop].get<double>());
        } else {
            reach.manning_n = Real(config_.default_manning_n);
        }
        
        // Read connectivity
        if (props.contains(config_.from_node_prop)) {
            reach_from_node[reach.id] = props[config_.from_node_prop].get<int>();
        }
        if (props.contains(config_.to_node_prop)) {
            reach_to_node[reach.id] = props[config_.to_node_prop].get<int>();
        }
        
        network.add_reach(reach);
    }
    
    // Build junction topology
    build_junctions(network, reach_from_node, reach_to_node);
    
    // Compute topological order
    network.build_topology();
    
    return network;
}

inline Network NetworkIO::load_csv(const std::string& reaches_csv,
                                    const std::string& params_csv) {
    Network network;
    
    std::ifstream file(reaches_csv);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open reaches CSV: " + reaches_csv);
    }
    
    std::unordered_map<int, int> reach_from_node;
    std::unordered_map<int, int> reach_to_node;
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        
        Reach reach;
        
        // reach_id
        std::getline(iss, token, ',');
        reach.id = std::stoi(token);
        reach.name = "reach_" + std::to_string(reach.id);
        
        // length_m
        std::getline(iss, token, ',');
        reach.length = std::stod(token);
        
        // slope
        std::getline(iss, token, ',');
        reach.slope = std::max(std::stod(token), config_.min_slope);
        
        // from_node
        std::getline(iss, token, ',');
        int from_node = std::stoi(token);
        if (from_node >= 0) {
            reach_from_node[reach.id] = from_node;
        }
        
        // to_node
        std::getline(iss, token, ',');
        int to_node = std::stoi(token);
        if (to_node >= 0) {
            reach_to_node[reach.id] = to_node;
        }
        
        // manning_n (optional)
        if (std::getline(iss, token, ',')) {
            reach.manning_n = Real(std::stod(token));
        } else {
            reach.manning_n = Real(config_.default_manning_n);
        }
        
        network.add_reach(reach);
    }
    
    // Load additional parameters if provided
    if (!params_csv.empty()) {
        std::ifstream params_file(params_csv);
        if (params_file.is_open()) {
            std::getline(params_file, line);  // Skip header
            
            while (std::getline(params_file, line)) {
                std::istringstream iss(line);
                std::string token;
                
                std::getline(iss, token, ',');
                int reach_id = std::stoi(token);
                
                try {
                    Reach& reach = network.get_reach(reach_id);
                    
                    if (std::getline(iss, token, ',') && !token.empty()) {
                        reach.manning_n = Real(std::stod(token));
                    }
                    if (std::getline(iss, token, ',') && !token.empty()) {
                        reach.geometry.width_coef = Real(std::stod(token));
                    }
                    if (std::getline(iss, token, ',') && !token.empty()) {
                        reach.geometry.width_exp = Real(std::stod(token));
                    }
                    if (std::getline(iss, token, ',') && !token.empty()) {
                        reach.geometry.depth_coef = Real(std::stod(token));
                    }
                    if (std::getline(iss, token, ',') && !token.empty()) {
                        reach.geometry.depth_exp = Real(std::stod(token));
                    }
                } catch (...) {
                    // Reach not found, skip
                }
            }
        }
    }
    
    build_junctions(network, reach_from_node, reach_to_node);
    network.build_topology();
    
    return network;
}

inline void NetworkIO::build_junctions(Network& network,
                                        const std::unordered_map<int, int>& reach_from_node,
                                        const std::unordered_map<int, int>& reach_to_node) {
    // Collect all unique node IDs
    std::set<int> node_ids;
    for (const auto& [reach_id, node_id] : reach_from_node) {
        node_ids.insert(node_id);
    }
    for (const auto& [reach_id, node_id] : reach_to_node) {
        node_ids.insert(node_id);
    }
    
    // Create junctions
    for (int node_id : node_ids) {
        Junction junc;
        junc.id = node_id;
        junc.name = "junction_" + std::to_string(node_id);
        
        // Find reaches that start at this node (upstream reaches)
        for (const auto& [reach_id, from_node] : reach_from_node) {
            if (from_node == node_id) {
                junc.downstream_reach_ids.push_back(reach_id);
            }
        }
        
        // Find reaches that end at this node (downstream reaches)
        for (const auto& [reach_id, to_node] : reach_to_node) {
            if (to_node == node_id) {
                junc.upstream_reach_ids.push_back(reach_id);
            }
        }
        
        junc.is_headwater = junc.upstream_reach_ids.empty();
        junc.is_outlet = junc.downstream_reach_ids.empty();
        
        network.add_junction(junc);
    }
    
    // Update reach junction references
    // (This requires access to reaches - we'll need to iterate)
    // The junction IDs are set, but we need to link reaches to them
    // This is handled during topological sorting
}

inline void NetworkIO::save_geojson(const Network& network, const std::string& filepath) {
    json geojson;
    geojson["type"] = "FeatureCollection";
    geojson["features"] = json::array();
    
    for (int reach_id : network.topological_order()) {
        const Reach& reach = network.get_reach(reach_id);
        
        json feature;
        feature["type"] = "Feature";
        feature["geometry"]["type"] = "LineString";
        feature["geometry"]["coordinates"] = json::array();  // Empty coords for now
        
        feature["properties"]["reach_id"] = reach.id;
        feature["properties"]["name"] = reach.name;
        feature["properties"]["length_m"] = reach.length;
        feature["properties"]["slope"] = reach.slope;
        feature["properties"]["manning_n"] = to_double(reach.manning_n);
        feature["properties"]["width_coef"] = to_double(reach.geometry.width_coef);
        feature["properties"]["width_exp"] = to_double(reach.geometry.width_exp);
        
        geojson["features"].push_back(feature);
    }
    
    std::ofstream file(filepath);
    file << geojson.dump(2);
}

inline void NetworkIO::save_csv(const Network& network, const std::string& filepath) {
    std::ofstream file(filepath);
    
    file << "reach_id,name,length_m,slope,manning_n,width_coef,width_exp,depth_coef,depth_exp\n";
    
    for (int reach_id : network.topological_order()) {
        const Reach& reach = network.get_reach(reach_id);
        
        file << reach.id << ","
             << reach.name << ","
             << reach.length << ","
             << reach.slope << ","
             << to_double(reach.manning_n) << ","
             << to_double(reach.geometry.width_coef) << ","
             << to_double(reach.geometry.width_exp) << ","
             << to_double(reach.geometry.depth_coef) << ","
             << to_double(reach.geometry.depth_exp) << "\n";
    }
}

} // namespace dmc

#endif // DMC_ROUTE_NETWORK_IO_HPP
