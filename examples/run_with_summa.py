#!/usr/bin/env python3
"""
Example: Running dMC-Route with SUMMA outputs

This script demonstrates how to:
1. Load a river network from NHDPlus GeoJSON
2. Read runoff from SUMMA NetCDF output files
3. Create HRU-to-reach mapping
4. Run routing and get gradients

Requirements:
- dmc_route compiled with NetCDF support
- SUMMA output files
- River network file
"""

import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Paths (adjust for your setup)
SUMMA_OUTPUT = "summa_output/domain_timestep.nc"
NETWORK_FILE = "network/flowlines.geojson"
MAPPING_FILE = "network/hru_reach_mapping.csv"
OUTPUT_FILE = "output/routed_discharge.csv"

# dMC-Route executable
DMC_ROUTE_BIN = "./build/dmc_route_run"

# ============================================================================
# Step 1: Inspect SUMMA Output
# ============================================================================

def inspect_summa_output(nc_file):
    """Print information about SUMMA output file."""
    try:
        import netCDF4 as nc
    except ImportError:
        print("netCDF4 not installed. Using ncdump instead.")
        subprocess.run(["ncdump", "-h", nc_file])
        return
    
    with nc.Dataset(nc_file) as ds:
        print("=== SUMMA Output Structure ===")
        print(f"\nDimensions:")
        for dim_name, dim in ds.dimensions.items():
            print(f"  {dim_name}: {len(dim)}")
        
        print(f"\nVariables:")
        for var_name, var in ds.variables.items():
            print(f"  {var_name}: {var.dimensions} {var.dtype}")
            if hasattr(var, 'units'):
                print(f"    units: {var.units}")
        
        # Look for runoff variables
        runoff_vars = [v for v in ds.variables if 'runoff' in v.lower() or 'flow' in v.lower()]
        print(f"\nPotential runoff variables: {runoff_vars}")

# ============================================================================
# Step 2: Create HRU-to-Reach Mapping
# ============================================================================

def create_mapping_from_nhdplus(flowlines_geojson, summa_nc, output_csv):
    """
    Create mapping between SUMMA HRUs and NHDPlus reaches.
    
    This assumes HRU IDs in SUMMA correspond to COMID in NHDPlus.
    Adjust the logic based on your specific setup.
    """
    try:
        import netCDF4 as nc
    except ImportError:
        print("Cannot create mapping without netCDF4")
        return
    
    # Read HRU IDs from SUMMA
    with nc.Dataset(summa_nc) as ds:
        if 'hruId' in ds.variables:
            hru_ids = ds.variables['hruId'][:].tolist()
        else:
            # Generate sequential IDs
            nhru = len(ds.dimensions['hru'])
            hru_ids = list(range(nhru))
    
    # Read reach IDs from network
    with open(flowlines_geojson) as f:
        network = json.load(f)
    
    reach_ids = []
    for feature in network['features']:
        if 'COMID' in feature['properties']:
            reach_ids.append(feature['properties']['COMID'])
    
    # Create mapping
    # Simple case: 1:1 mapping where HRU ID == reach ID
    mapping = []
    for reach_id in reach_ids:
        if reach_id in hru_ids:
            hru_idx = hru_ids.index(reach_id)
            mapping.append({
                'reach_id': reach_id,
                'hru_id': reach_id,
                'weight': 1.0
            })
    
    # Save mapping
    df = pd.DataFrame(mapping)
    df.to_csv(output_csv, index=False)
    print(f"Created mapping with {len(mapping)} entries -> {output_csv}")
    
    return df

# ============================================================================
# Step 3: Run dMC-Route
# ============================================================================

def run_routing(network_file, forcing_file, mapping_file, output_file, 
                enable_gradients=True):
    """Run dMC-Route executable."""
    
    cmd = [
        DMC_ROUTE_BIN,
        "--network", network_file,
        "--forcing", forcing_file,
        "--output", output_file,
    ]
    
    if mapping_file and Path(mapping_file).exists():
        cmd.extend(["--mapping", mapping_file])
    
    if enable_gradients:
        cmd.append("--gradients")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

# ============================================================================
# Step 4: Analyze Results
# ============================================================================

def plot_results(output_file, obs_file=None):
    """Plot routing results and optionally compare to observations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return
    
    df = pd.read_csv(output_file)
    
    # Get discharge columns (Q_*)
    q_cols = [c for c in df.columns if c.startswith('Q_')]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot outlet (last reach)
    outlet_col = q_cols[-1]
    ax.plot(df['time'] / 3600, df[outlet_col], 'b-', label='Simulated')
    
    if obs_file and Path(obs_file).exists():
        obs = pd.read_csv(obs_file)
        ax.plot(obs['time'] / 3600, obs['discharge'], 'k.', 
                label='Observed', alpha=0.5)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title(f'Routed Discharge at {outlet_col}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file.replace('.csv', '.png'), dpi=150)
    print(f"Saved plot to {output_file.replace('.csv', '.png')}")

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("dMC-Route + SUMMA Integration Example")
    print("=" * 60)
    
    # Check files exist
    if not Path(SUMMA_OUTPUT).exists():
        print(f"\nSUMMA output not found: {SUMMA_OUTPUT}")
        print("Using demo mode instead...")
        subprocess.run([DMC_ROUTE_BIN, "--demo"])
        return
    
    # Step 1: Inspect SUMMA output
    print("\n[Step 1] Inspecting SUMMA output...")
    inspect_summa_output(SUMMA_OUTPUT)
    
    # Step 2: Create HRU-reach mapping
    print("\n[Step 2] Creating HRU-reach mapping...")
    if Path(NETWORK_FILE).exists():
        create_mapping_from_nhdplus(NETWORK_FILE, SUMMA_OUTPUT, MAPPING_FILE)
    else:
        print(f"Network file not found: {NETWORK_FILE}")
    
    # Step 3: Run routing
    print("\n[Step 3] Running routing...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    success = run_routing(
        NETWORK_FILE, 
        SUMMA_OUTPUT, 
        MAPPING_FILE, 
        OUTPUT_FILE,
        enable_gradients=True
    )
    
    if success:
        # Step 4: Analyze results
        print("\n[Step 4] Analyzing results...")
        plot_results(OUTPUT_FILE)
        print("\nDone!")
    else:
        print("\nRouting failed!")

if __name__ == "__main__":
    main()
