#!/usr/bin/env python3
"""
Test dRoute routing methods with real SUMMA simulation data and observed streamflow.

This script:
1. Loads SUMMA runoff simulations and observed streamflow
2. Runs each routing method (MC, Lag, IRF, KWT)
3. Computes performance metrics (NSE, KGE, RMSE)
4. Optionally optimizes parameters using PyTorch (with CoDiPack AD)
5. Plots results comparing methods and before/after optimization

Usage:
    python test_routing_with_data.py --data-dir /path/to/data
    python test_routing_with_data.py --optimize
    python test_routing_with_data.py --optimize --fast  # Use numerical gradients (faster)
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Try to import optional dependencies
try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False
    print("Warning: xarray not found. Install with: pip install xarray netcdf4")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. Install with: pip install torch")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")

# Import dRoute bindings
try:
    import droute
    HAS_DROUTE = True
    print(f"Loaded droute version {droute.__version__}")
except ImportError:
    HAS_DROUTE = False
    print("Warning: droute not found. Build with: pip install -e .")


# =============================================================================
# Performance Metrics
# =============================================================================

def nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    obs_mean = np.nanmean(obs)
    numerator = np.nansum((obs - sim) ** 2)
    denominator = np.nansum((obs - obs_mean) ** 2)
    return 1.0 - numerator / denominator if denominator > 0 else -np.inf


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    """Kling-Gupta Efficiency."""
    # Remove NaN pairs
    mask = ~(np.isnan(sim) | np.isnan(obs))
    sim, obs = sim[mask], obs[mask]
    
    if len(sim) < 2:
        return -np.inf
    
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else 0
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else 0
    
    return 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def rmse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Root Mean Square Error."""
    return np.sqrt(np.nanmean((sim - obs) ** 2))


def pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    """Percent Bias."""
    obs_sum = np.nansum(obs)
    return 100.0 * np.nansum(sim - obs) / obs_sum if obs_sum != 0 else 0.0


def load_topology(filepath: Path) -> Tuple['droute.Network', np.ndarray, int, Dict[int, int]]:
    """
    Load river network topology from mizuRoute topology.nc file.
    
    Args:
        filepath: Path to topology.nc
        
    Returns:
        network: Configured droute.Network with proper topology connections
        seg_areas: Array of HRU areas in m² (indexed by reach index)
        outlet_idx: Index of outlet reach
        hru_to_seg_idx: Mapping from HRU ID to segment index
    """
    if not HAS_XARRAY:
        raise ImportError("xarray required to load NetCDF files")
    if not HAS_DROUTE:
        raise ImportError("droute required")
    
    ds = xr.open_dataset(filepath)
    
    # Extract data
    seg_ids = ds['segId'].values
    down_seg_ids = ds['downSegId'].values
    slopes = ds['slope'].values
    lengths = ds['length'].values
    mann_n = ds['mann_n'].values
    
    hru_ids = ds['hruId'].values
    hru_to_seg = ds['hruToSegId'].values
    hru_areas = ds['area'].values  # m²
    
    ds.close()
    
    n_segs = len(seg_ids)
    
    # Create segment ID to index mapping
    seg_id_to_idx = {int(seg_id): i for i, seg_id in enumerate(seg_ids)}
    
    # Create HRU ID to segment index mapping (for reordering runoff data)
    hru_to_seg_idx = {}
    for i, hru_id in enumerate(hru_ids):
        seg_id = int(hru_to_seg[i])
        if seg_id in seg_id_to_idx:
            hru_to_seg_idx[int(hru_id)] = seg_id_to_idx[seg_id]
    
    # Find outlet (segment whose downstream doesn't exist in seg_ids)
    outlet_idx = 0
    for i, down_id in enumerate(down_seg_ids):
        if int(down_id) not in seg_id_to_idx:
            outlet_idx = i
            break
    
    # Build upstream connectivity map: downstream_idx -> [upstream_idx, ...]
    upstream_map = {i: [] for i in range(n_segs)}
    for i, down_id in enumerate(down_seg_ids):
        down_id_int = int(down_id)
        if down_id_int in seg_id_to_idx:
            down_idx = seg_id_to_idx[down_id_int]
            upstream_map[down_idx].append(i)
    
    # Build network
    network = droute.Network()
    
    # Create all reaches with proper junction IDs
    # Convention: each reach has a junction at its upstream end with same ID
    for i in range(n_segs):
        reach = droute.Reach()
        reach.id = i
        reach.length = float(lengths[i])
        reach.slope = max(float(slopes[i]), 0.0001)
        reach.manning_n = float(mann_n[i])
        
        # Default geometry (power law)
        reach.geometry.width_coef = 7.2
        reach.geometry.width_exp = 0.5
        reach.geometry.depth_coef = 0.27
        reach.geometry.depth_exp = 0.3
        
        # Junction at upstream end of this reach
        reach.upstream_junction_id = i
        
        # Junction at downstream end = upstream junction of downstream reach
        down_id = int(down_seg_ids[i])
        if down_id in seg_id_to_idx:
            reach.downstream_junction_id = seg_id_to_idx[down_id]
        else:
            reach.downstream_junction_id = -1  # Outlet
        
        network.add_reach(reach)
    
    # Create junctions - one at upstream end of each reach
    for i in range(n_segs):
        junc = droute.Junction()
        junc.id = i
        
        # Upstream reaches that flow into this junction
        junc.upstream_reach_ids = upstream_map[i]
        
        # This junction feeds into reach i
        junc.downstream_reach_ids = [i]
        
        network.add_junction(junc)
    
    network.build_topology()
    
    # Verify topological order
    topo_order = network.topological_order()
    
    # Create area array indexed by reach index
    seg_areas = np.zeros(n_segs)
    for i, hru_id in enumerate(hru_ids):
        seg_id = int(hru_to_seg[i])
        if seg_id in seg_id_to_idx:
            seg_idx = seg_id_to_idx[seg_id]
            seg_areas[seg_idx] = hru_areas[i]
    
    # Count upstream connections
    n_headwaters = sum(1 for v in upstream_map.values() if not v)
    n_confluences = sum(1 for v in upstream_map.values() if len(v) > 1)
    
    print(f"  Loaded topology: {n_segs} segments")
    print(f"    Headwater reaches: {n_headwaters}")
    print(f"    Confluences: {n_confluences}")
    print(f"    Topological order: {len(topo_order)} reaches")
    print(f"  Total basin area: {seg_areas.sum()/1e6:.1f} km²")
    print(f"  Reach lengths: {lengths.min():.0f} - {lengths.max():.0f} m")
    print(f"  Slopes: {slopes.min():.6f} - {slopes.max():.4f}")
    print(f"  Manning's n: {mann_n.min():.4f} - {mann_n.max():.4f}")
    print(f"  Outlet: segment {seg_ids[outlet_idx]} (index {outlet_idx})")
    
    return network, seg_areas, outlet_idx, hru_to_seg_idx


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_hydrographs(times: np.ndarray, 
                     observed: np.ndarray,
                     simulations: Dict[str, np.ndarray],
                     title: str = "Routing Method Comparison",
                     filename: str = "hydrograph_comparison.png"):
    """
    Plot observed vs simulated hydrographs for multiple methods.
    
    Args:
        times: Time array (can be datetime or numeric)
        observed: Observed discharge
        simulations: Dict of {method_name: simulated_discharge}
        title: Plot title
        filename: Output filename
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot {filename} - matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Color scheme
    colors = {
        'observed': 'black',
        'mc': '#1f77b4',
        'lag': '#ff7f0e', 
        'irf': '#2ca02c',
        'kwt': '#d62728',
        'diffusive': '#9467bd',
        'sve': '#8c564b',  # Brown for Saint-Venant
    }
    
    # Top panel: All hydrographs
    ax1 = axes[0]
    ax1.plot(times, observed, 'k-', linewidth=2, label='Observed', alpha=0.8)
    
    for method, sim in simulations.items():
        color = colors.get(method.lower().split('_')[0], 'gray')
        linestyle = '--' if 'optimized' in method.lower() else '-'
        ax1.plot(times, sim, color=color, linestyle=linestyle, 
                linewidth=1.5, label=method, alpha=0.7)
    
    ax1.set_ylabel('Discharge (m³/s)')
    ax1.set_title(title)
    ax1.legend(loc='upper right', ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Residuals
    ax2 = axes[1]
    for method, sim in simulations.items():
        color = colors.get(method.lower().split('_')[0], 'gray')
        linestyle = '--' if 'optimized' in method.lower() else '-'
        residual = sim - observed
        ax2.plot(times, residual, color=color, linestyle=linestyle,
                linewidth=1, alpha=0.7, label=method)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Residual (m³/s)')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {filename}")


def plot_optimization_comparison(times: np.ndarray,
                                  observed: np.ndarray,
                                  default_sim: np.ndarray,
                                  optimized_sim: np.ndarray,
                                  method: str,
                                  metrics_default: Dict,
                                  metrics_optimized: Dict,
                                  filename: str = None):
    """
    Plot before/after optimization comparison for a single method.
    """
    if not HAS_MATPLOTLIB:
        return
    
    if filename is None:
        filename = f"optimization_{method}.png"
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top: Hydrographs
    ax1 = axes[0]
    ax1.plot(times, observed, 'k-', linewidth=2, label='Observed', alpha=0.8)
    ax1.plot(times, default_sim, 'b--', linewidth=1.5, 
             label=f'Default (NSE={metrics_default["nse"]:.2f})', alpha=0.7)
    ax1.plot(times, optimized_sim, 'r-', linewidth=1.5,
             label=f'Optimized (NSE={metrics_optimized["nse"]:.2f})', alpha=0.7)
    
    ax1.set_ylabel('Discharge (m³/s)')
    ax1.set_title(f'{method.upper()} Routing: Before vs After Optimization')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Improvement in residuals
    ax2 = axes[1]
    ax2.fill_between(times, default_sim - observed, 0, 
                     alpha=0.3, color='blue', label='Default residual')
    ax2.fill_between(times, optimized_sim - observed, 0,
                     alpha=0.3, color='red', label='Optimized residual')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Residual (m³/s)')
    ax2.set_xlabel('Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {filename}")


# =============================================================================
# Data Loading
# =============================================================================

def load_summa_runoff(filepath: Path, quiet: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, Optional[np.ndarray]]:
    """
    Load SUMMA simulation output.
    
    Args:
        filepath: Path to SUMMA NetCDF file
        quiet: If True, suppress warnings about missing area variables
        
    Returns:
        runoff: Array of shape (n_timesteps, n_gru) with runoff in m/s
        gru_ids: Array of GRU IDs
        times: DatetimeIndex of timesteps
        areas: Array of GRU areas in m² (or None if not found)
    """
    if not HAS_XARRAY:
        raise ImportError("xarray required to load NetCDF files")
    
    ds = xr.open_dataset(filepath)
    
    # Get runoff - averageRoutedRunoff is in m/s
    runoff = ds['averageRoutedRunoff'].values  # (time, gru)
    gru_ids = ds['gruId'].values
    
    # Convert time - handle both datetime64 and numeric formats
    time_vals = ds['time'].values
    if np.issubdtype(time_vals.dtype, np.datetime64):
        # Already datetime64
        times = pd.DatetimeIndex(time_vals)
    else:
        # Numeric - parse units from attributes
        time_units = ds['time'].attrs.get('units', 's since 1990-1-1 0:0:0')
        # Extract origin from units string like "s since 1990-1-1 0:0:0"
        if 'since' in time_units:
            origin_str = time_units.split('since')[1].strip()
            times = pd.to_datetime(time_vals, unit='s', origin=origin_str)
        else:
            times = pd.to_datetime(time_vals, unit='s', origin='1990-01-01')
    
    # Try to get GRU areas (various possible variable names)
    areas = None
    area_var_names = ['HRUarea', 'hruArea', 'gruArea', 'area', 'HRU_area', 'GRU_area']
    for var_name in area_var_names:
        if var_name in ds.variables:
            areas = ds[var_name].values
            print(f"  Found GRU areas in variable '{var_name}'")
            print(f"    Total area: {areas.sum()/1e6:.1f} km²")
            print(f"    Mean GRU area: {areas.mean()/1e6:.2f} km²")
            break
    
    if areas is None and not quiet:
        print("  Note: No GRU area variable in SUMMA file (will use topology areas)")
    
    ds.close()
    
    return runoff, gru_ids, times, areas


def load_observations(filepath: Path) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Load observed streamflow data.
    
    Args:
        filepath: Path to CSV file with datetime and discharge_cms columns
        
    Returns:
        times: DatetimeIndex
        discharge: Array of discharge values in m³/s
    """
    df = pd.read_csv(filepath, parse_dates=['datetime'])
    df = df.set_index('datetime')
    return df.index, df['discharge_cms'].values


def create_synthetic_network(n_reaches: int = 10, 
                             reach_length: float = 5000.0,
                             slope: float = 0.001) -> 'droute.Network':
    """
    Create a simple linear synthetic network for testing.
    
    Args:
        n_reaches: Number of reaches (linear chain)
        reach_length: Length of each reach in meters
        slope: Bed slope
        
    Returns:
        Network object
    """
    if not HAS_DROUTE:
        raise ImportError("droute required")
    
    network = droute.Network()
    
    # Create linear chain of reaches
    for i in range(n_reaches):
        reach = droute.Reach()
        reach.id = i
        reach.name = f"reach_{i}"
        reach.length = reach_length
        reach.slope = slope
        reach.manning_n = 0.035
        
        # Power-law geometry coefficients
        reach.geometry.width_coef = 7.2
        reach.geometry.width_exp = 0.5
        reach.geometry.depth_coef = 0.27
        reach.geometry.depth_exp = 0.3
        
        network.add_reach(reach)
    
    network.build_topology()
    return network


# =============================================================================
# Routing Tests
# =============================================================================

ROUTER_CLASSES = {
    'mc': 'MuskingumCungeRouter',
    'lag': 'LagRouter',
    'irf': 'IRFRouter',
    'kwt': 'SoftGatedKWT',
    'diffusive': 'DiffusiveWaveIFT',
    'sve': 'SaintVenantRouter',  # Full dynamic Saint-Venant
}


class RoutingTest:
    """Test harness for routing methods."""
    
    def __init__(self, network: 'droute.Network', 
                 runoff: np.ndarray,
                 dt: float = 3600.0,
                 outlet_reach: int = None):
        """
        Args:
            network: River network
            runoff: Lateral inflows (n_timesteps, n_reaches) in m³/s
            dt: Timestep in seconds
            outlet_reach: Index of outlet reach (default: last reach)
        """
        self.network = network
        self.runoff = runoff
        self.dt = dt
        self.n_timesteps = runoff.shape[0]
        self.n_reaches = runoff.shape[1]
        self.outlet_reach = outlet_reach if outlet_reach is not None else self._find_outlet_reach()
        
    def run_method(self, method: str, **kwargs) -> np.ndarray:
        """
        Run a routing method and return outlet discharge timeseries.
        
        Args:
            method: One of 'mc', 'lag', 'irf', 'kwt', 'diffusive', 'sve'
            **kwargs: Method-specific parameters
            
        Returns:
            Array of outlet discharge (n_timesteps,)
        """
        if not HAS_DROUTE:
            raise ImportError("droute required")
        
        # Create router config based on method
        if method == 'sve':
            # Saint-Venant uses its own config class
            config = droute.SaintVenantConfig()
            config.dt = self.dt
            config.n_nodes = kwargs.get('n_nodes', 10)
            config.initial_depth = kwargs.get('initial_depth', 0.5)
            config.initial_velocity = kwargs.get('initial_velocity', 0.1)
        else:
            config = droute.RouterConfig()
            config.dt = self.dt
            config.enable_gradients = kwargs.get('enable_gradients', False)
            
            # Method-specific config
            if method == 'mc':
                config.num_substeps = kwargs.get('num_substeps', 4)
            elif method == 'irf':
                config.irf_shape_param = kwargs.get('shape_param', 2.5)
                config.irf_max_kernel_size = kwargs.get('kernel_size', 100)
            elif method == 'kwt':
                config.kwt_gate_steepness = kwargs.get('gate_steepness', 5.0)
        
        # Create router
        router_class_name = ROUTER_CLASSES.get(method)
        if not router_class_name:
            raise ValueError(f"Unknown method: {method}")
        
        RouterClass = getattr(droute, router_class_name)
        router = RouterClass(self.network, config)
        
        # Run simulation
        outlet_Q = np.zeros(self.n_timesteps)
        
        start_time = time.time()
        
        for t in range(self.n_timesteps):
            # Set lateral inflows for all reaches
            for r in range(self.n_reaches):
                router.set_lateral_inflow(r, float(self.runoff[t, r]))
            
            # Route
            router.route_timestep()
            
            # Get outlet discharge
            outlet_Q[t] = router.get_discharge(self.outlet_reach)
        
        elapsed = time.time() - start_time
        print(f"  {method.upper():12s}: {self.n_timesteps} timesteps in {elapsed:.3f}s "
              f"({self.n_timesteps/elapsed:.0f} timesteps/s)")
        
        return outlet_Q
    
    def _find_outlet_reach(self) -> int:
        """Find the outlet reach (one with downstream_id = -1)."""
        for i in range(self.n_reaches):
            try:
                reach = self.network.get_reach(i)
                if reach.downstream_id == -1:
                    return i
            except:
                pass
        # Fallback to last reach
        return self.n_reaches - 1


# =============================================================================
# PyTorch Optimization (using CoDiPack AD through dRoute)
# =============================================================================

def optimize_routing_pytorch(network: 'droute.Network',
                             runoff: np.ndarray,
                             observed: np.ndarray,
                             method: str = 'mc',
                             n_epochs: int = 50,
                             lr: float = 0.01,
                             dt: float = 3600.0,
                             outlet_reach: int = None) -> Dict:
    """
    Optimize routing parameters using PyTorch Adam optimizer with CoDiPack AD.
    
    Uses dRoute's CoDiPack-based routers with proper timeseries gradient
    accumulation. The record_output() method stores each timestep's discharge
    on the AD tape, and compute_gradients_timeseries() backprops through all
    timesteps in a single reverse pass.
    
    Args:
        network: River network
        runoff: Lateral inflows (n_timesteps, n_reaches)
        observed: Observed outlet discharge (n_timesteps,)
        method: Routing method ('mc', 'lag', 'irf', 'kwt', 'diffusive')
        n_epochs: Number of optimization epochs
        lr: Learning rate
        dt: Timestep in seconds
        outlet_reach: Index of outlet reach (auto-detected if None)
        
    Returns:
        Dictionary with optimized parameters and final metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for optimization")
    if not HAS_DROUTE:
        raise ImportError("droute required")
    
    n_timesteps = runoff.shape[0]
    n_reaches = runoff.shape[1]
    
    # Find outlet if not specified
    if outlet_reach is None:
        outlet_reach = 0
        for i in range(n_reaches):
            reach = network.get_reach(i)
            if reach.downstream_junction_id < 0:
                outlet_reach = i
                break
    
    # Create config with gradients enabled
    config = droute.RouterConfig()
    config.dt = dt
    config.enable_gradients = True
    
    # Get router class
    RouterClass = getattr(droute, ROUTER_CLASSES[method])
    
    # Check if router supports timeseries gradients
    has_timeseries_grad = hasattr(RouterClass, '__init__') and method == 'mc'
    # TODO: Add timeseries gradient support to other routers
    
    # Initialize learnable parameters in PyTorch
    log_manning_n = torch.tensor(
        [np.log(max(network.get_reach(i).manning_n, 0.01)) 
         for i in range(n_reaches)],
        dtype=torch.float64,
        requires_grad=True
    )
    
    # PyTorch Adam optimizer
    optimizer = torch.optim.Adam([log_manning_n], lr=lr)
    
    print(f"\nOptimizing {method.upper()} routing...")
    print(f"  {n_reaches} reaches, {n_timesteps} timesteps")
    if has_timeseries_grad:
        print(f"  Using CoDiPack timeseries AD gradients")
    else:
        print(f"  Using numerical gradients (CoDiPack timeseries not available for {method})")
    
    losses = []
    start_time = time.time()
    
    # Finite difference step size (for non-MC methods)
    eps_fd = 0.01
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # Update network parameters from PyTorch tensor
        manning_n = torch.exp(log_manning_n)
        for i in range(n_reaches):
            network.get_reach(i).manning_n = float(manning_n[i].item())
        
        if has_timeseries_grad:
            # === CoDiPack AD Path ===
            # Create fresh router with AD enabled
            router = RouterClass(network, config)
            router.start_recording()
            
            # Forward pass - record output at each timestep
            sim = np.zeros(n_timesteps)
            for t in range(n_timesteps):
                for r in range(n_reaches):
                    router.set_lateral_inflow(r, float(runoff[t, r]))
                router.route_timestep()
                router.record_output(outlet_reach)  # Record for AD
                sim[t] = router.get_discharge(outlet_reach)
            
            router.stop_recording()
            
            # Compute MSE loss
            mse = np.mean((sim - observed) ** 2)
            losses.append(mse)
            
            # Compute dL/dQ for each timestep: dMSE/dQ_t = 2*(sim_t - obs_t)/T
            dL_dQ = (2.0 / n_timesteps) * (sim - observed)
            
            # Backprop through entire timeseries
            router.compute_gradients_timeseries(outlet_reach, dL_dQ.tolist())
            
            # Get gradients from CoDiPack
            grads = router.get_gradients()
            
            # Transfer to PyTorch
            grad_manning = np.array([
                grads.get(f"reach_{i}_manning_n", 0.0) for i in range(n_reaches)
            ])
            
            # Chain rule for log transform: d/d(log_n) = d/d(n) * n
            log_manning_n.grad = torch.tensor(
                grad_manning * manning_n.detach().numpy(),
                dtype=torch.float64
            )
            
        else:
            # === Numerical Gradient Path (for non-MC methods) ===
            def run_sim(log_n_np):
                for i in range(n_reaches):
                    network.get_reach(i).manning_n = float(np.exp(log_n_np[i]))
                router = RouterClass(network, config)
                sim = np.zeros(n_timesteps)
                for t in range(n_timesteps):
                    for r in range(n_reaches):
                        router.set_lateral_inflow(r, float(runoff[t, r]))
                    router.route_timestep()
                    sim[t] = router.get_discharge(outlet_reach)
                return np.mean((sim - observed) ** 2), sim
            
            log_n_np = log_manning_n.detach().numpy().copy()
            base_loss, sim = run_sim(log_n_np)
            losses.append(base_loss)
            
            # Numerical gradients
            grad_np = np.zeros(n_reaches)
            for i in range(n_reaches):
                log_n_np[i] += eps_fd
                loss_plus, _ = run_sim(log_n_np)
                log_n_np[i] -= 2 * eps_fd
                loss_minus, _ = run_sim(log_n_np)
                log_n_np[i] += eps_fd
                grad_np[i] = (loss_plus - loss_minus) / (2 * eps_fd)
            
            log_manning_n.grad = torch.tensor(grad_np, dtype=torch.float64)
            mse = base_loss
        
        # Adam update step
        optimizer.step()
        
        # Clamp to reasonable range
        with torch.no_grad():
            log_manning_n.clamp_(np.log(0.01), np.log(0.2))
        
        # Progress output
        current_nse = nse(sim, observed)
        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: MSE = {mse:.4f}, "
                  f"NSE = {current_nse:.3f}")
    
    elapsed = time.time() - start_time
    
    # Final simulation with optimized parameters
    final_manning = torch.exp(log_manning_n).detach().numpy()
    for i in range(n_reaches):
        network.get_reach(i).manning_n = float(final_manning[i])
    
    config_final = droute.RouterConfig()
    config_final.dt = dt
    config_final.enable_gradients = False
    router = RouterClass(network, config_final)
    
    sim_final = np.zeros(n_timesteps)
    for t in range(n_timesteps):
        for r in range(n_reaches):
            router.set_lateral_inflow(r, float(runoff[t, r]))
        router.route_timestep()
        sim_final[t] = router.get_discharge(outlet_reach)
    
    # Results
    results = {
        'method': method,
        'nse': nse(sim_final, observed),
        'kge': kge(sim_final, observed),
        'rmse': rmse(sim_final, observed),
        'pbias': pbias(sim_final, observed),
        'training_time': elapsed,
        'final_loss': losses[-1] if losses else 0.0,
        'optimized_manning_n': final_manning,
        'simulated': sim_final,
        'losses': np.array(losses),
    }
    
    return results


def optimize_routing_fast(network: 'droute.Network',
                          runoff: np.ndarray,
                          observed: np.ndarray,
                          method: str = 'mc',
                          n_epochs: int = 30,
                          lr: float = 0.1,
                          dt: float = 3600.0) -> Dict:
    """
    Fast optimization using numerical gradients (finite differences).
    
    This is faster than CoDiPack tape-based AD for small networks because
    it avoids the overhead of tape recording.
    
    Args:
        network: River network
        runoff: Lateral inflows (n_timesteps, n_reaches)
        observed: Observed outlet discharge (n_timesteps,)
        method: Routing method
        n_epochs: Number of optimization epochs
        lr: Learning rate
        dt: Timestep in seconds
        
    Returns:
        Dictionary with optimized parameters and final metrics
    """
    if not HAS_DROUTE:
        raise ImportError("droute required")
    
    n_timesteps = runoff.shape[0]
    n_reaches = runoff.shape[1]
    outlet_reach = n_reaches - 1
    
    # Create config (no gradients needed - we'll use finite diff)
    config = droute.RouterConfig()
    config.dt = dt
    config.enable_gradients = False  # Faster without AD
    
    RouterClass = getattr(droute, ROUTER_CLASSES[method])
    
    # Initialize parameters
    manning_n = np.array([network.get_reach(i).manning_n for i in range(n_reaches)])
    log_manning_n = np.log(manning_n)
    
    def run_simulation(log_n_values):
        """Run simulation with given log(manning_n) values."""
        # Update network
        n_values = np.exp(log_n_values)
        for i in range(n_reaches):
            network.get_reach(i).manning_n = float(n_values[i])
        
        # Create fresh router
        router = RouterClass(network, config)
        
        # Run simulation
        sim = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            for r in range(n_reaches):
                router.set_lateral_inflow(r, float(runoff[t, r]))
            router.route_timestep()
            sim[t] = router.get_discharge(outlet_reach)
        
        return sim
    
    def compute_loss(log_n_values):
        """Compute MSE loss."""
        sim = run_simulation(log_n_values)
        return np.mean((sim - observed) ** 2), sim
    
    def compute_numerical_gradient(log_n_values, eps=0.01):
        """Compute gradient using central finite differences."""
        grad = np.zeros_like(log_n_values)
        base_loss, _ = compute_loss(log_n_values)
        
        for i in range(len(log_n_values)):
            log_n_plus = log_n_values.copy()
            log_n_plus[i] += eps
            loss_plus, _ = compute_loss(log_n_plus)
            
            log_n_minus = log_n_values.copy()
            log_n_minus[i] -= eps
            loss_minus, _ = compute_loss(log_n_minus)
            
            grad[i] = (loss_plus - loss_minus) / (2 * eps)
        
        return grad, base_loss
    
    print(f"\nOptimizing {method.upper()} routing (numerical gradients)...")
    print(f"  {n_reaches} reaches, {n_timesteps} timesteps")
    
    losses = []
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Compute gradient
        grad, loss = compute_numerical_gradient(log_manning_n)
        losses.append(loss)
        
        # Gradient descent update
        log_manning_n -= lr * grad
        
        # Clip to reasonable range
        log_manning_n = np.clip(log_manning_n, np.log(0.01), np.log(0.2))
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sim = run_simulation(log_manning_n)
            current_nse = nse(sim, observed)
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: MSE = {loss:.4f}, NSE = {current_nse:.3f}")
    
    elapsed = time.time() - start_time
    
    # Final simulation
    sim_final = run_simulation(log_manning_n)
    
    results = {
        'method': method,
        'nse': nse(sim_final, observed),
        'kge': kge(sim_final, observed),
        'rmse': rmse(sim_final, observed),
        'pbias': pbias(sim_final, observed),
        'training_time': elapsed,
        'final_loss': losses[-1],
        'optimized_manning_n': np.exp(log_manning_n),
        'simulated': sim_final,
    }
    
    return results


# =============================================================================
# Main Test Runner
# =============================================================================

def generate_synthetic_data(n_timesteps: int = 1000, 
                             n_reaches: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic runoff and observation data for testing."""
    # Synthetic runoff with temporal pattern
    t = np.arange(n_timesteps)
    
    # Base signal: sinusoidal + some peaks
    base_runoff = 5.0 + 3.0 * np.sin(2 * np.pi * t / 200)
    base_runoff += 5.0 * np.exp(-((t - 300) ** 2) / 1000)  # Event 1
    base_runoff += 8.0 * np.exp(-((t - 700) ** 2) / 2000)  # Event 2
    base_runoff = np.maximum(base_runoff, 1.0)
    
    # Spatial distribution across reaches (decreasing upstream to downstream)
    runoff_cms = np.outer(base_runoff, np.linspace(1.5, 0.5, n_reaches))
    
    # Synthetic observations (sum of upstream contributions, lagged and attenuated)
    # Simple convolution to simulate routing effect
    kernel = np.exp(-np.arange(20) / 5)
    kernel /= kernel.sum()
    observed = np.convolve(base_runoff * n_reaches * 0.8, kernel, mode='same')
    observed += np.random.normal(0, 0.5, n_timesteps)  # Add noise
    observed = np.maximum(observed, 0.1)
    
    return runoff_cms, observed


def run_tests(data_dir: Optional[Path] = None,
              optimize: bool = False,
              fast: bool = False,
              skip_years: int = 1,
              max_years: int = 1,
              methods: List[str] = ['mc', 'lag', 'irf', 'kwt', 'diffusive']) -> Dict:
    """
    Run routing tests with real or synthetic data.
    
    Args:
        data_dir: Path to data directory (optional)
        optimize: Whether to run optimization
        fast: Use fast numerical gradients instead of CoDiPack AD
        skip_years: Number of years to skip at start (spinup)
        max_years: Maximum years to simulate after spinup
        methods: Routing methods to test
        
    Returns:
        Dictionary with all results
    """
    results = {}
    times_array = None  # For plotting
    network = None
    seg_areas = None
    outlet_idx = None
    hru_to_seg_idx = None
    
    # Load or generate data
    if data_dir and Path(data_dir).exists():
        print(f"Loading data from {data_dir}...")
        data_dir = Path(data_dir)
        
        # Load topology if available
        topo_paths = list(data_dir.glob("**/topology.nc"))
        if topo_paths:
            topo_path = topo_paths[0]
            print(f"\n  Found topology file: {topo_path}")
            network, seg_areas, outlet_idx, hru_to_seg_idx = load_topology(topo_path)
            n_reaches = network.num_reaches()
        else:
            print("  No topology.nc found, will use synthetic network")
            n_reaches = None
        
        # Load SUMMA output
        summa_paths = list(data_dir.glob("**/SUMMA/*_timestep.nc"))
        if summa_paths:
            summa_path = summa_paths[0]
            print(f"\n  Found SUMMA file: {summa_path.name}")
            runoff, gru_ids, times_array, file_areas = load_summa_runoff(summa_path, 
                                                                          quiet=(seg_areas is not None))
            print(f"  Loaded runoff: {runoff.shape} (timesteps x GRUs)")
            
            if n_reaches is None:
                n_reaches = runoff.shape[1]
            
            # Reorder runoff from HRU order to segment order
            if hru_to_seg_idx is not None:
                print(f"  Reordering runoff from HRU order to segment order...")
                runoff_reordered = np.zeros((runoff.shape[0], n_reaches))
                for i, gru_id in enumerate(gru_ids):
                    gru_id_int = int(gru_id)
                    if gru_id_int in hru_to_seg_idx:
                        seg_idx = hru_to_seg_idx[gru_id_int]
                        runoff_reordered[:, seg_idx] = runoff[:, i]
                runoff = runoff_reordered
            
            # Convert from m/s to m³/s using topology areas
            if seg_areas is not None:
                print(f"  Using HRU areas from topology.nc")
                runoff_cms = runoff * seg_areas[np.newaxis, :]  # broadcast (time, seg)
            elif file_areas is not None:
                print(f"  Using HRU areas from SUMMA file")
                runoff_cms = runoff * file_areas[np.newaxis, :]
            else:
                # Fallback
                area_per_gru = 45e6  # m² (45 km²)
                print(f"  WARNING: No area info, using estimate: 45 km² per GRU")
                runoff_cms = runoff * area_per_gru
            
            # Apply spinup skip and max years
            hours_per_year = 365 * 24
            skip_hours = skip_years * hours_per_year
            max_hours = max_years * hours_per_year
            
            total_hours = runoff_cms.shape[0]
            start_idx = min(skip_hours, total_hours - 100)
            end_idx = min(start_idx + max_hours, total_hours)
            
            runoff_cms = runoff_cms[start_idx:end_idx]
            if times_array is not None:
                times_array = times_array[start_idx:end_idx]
            
            # Print runoff diagnostics
            print(f"\n  Skipped first {skip_years} year(s), using {(end_idx-start_idx)/hours_per_year:.1f} year(s)")
            print(f"  Timesteps: {runoff_cms.shape[0]}")
            if times_array is not None:
                print(f"  Date range: {times_array[0]} to {times_array[-1]}")
            print(f"  Runoff per GRU range: {runoff_cms.min():.4f} to {runoff_cms.max():.4f} m³/s")
            print(f"  Total runoff range: {runoff_cms.sum(axis=1).min():.2f} to {runoff_cms.sum(axis=1).max():.2f} m³/s")
        else:
            print("  No SUMMA file found, using synthetic data")
            n_reaches = 10
            runoff_cms, _ = generate_synthetic_data(1000, n_reaches)
        
        # Load observations
        obs_paths = list(data_dir.glob("**/observations/**/*.csv"))
        if obs_paths:
            obs_path = obs_paths[0]
            print(f"\n  Found observation file: {obs_path.name}")
            obs_times, observed_full = load_observations(obs_path)
            print(f"  Loaded observations: {len(observed_full)} timesteps")
            
            # Match observations to simulation time period
            if times_array is not None:
                # Find matching time range
                obs_start = times_array[0]
                obs_end = times_array[-1]
                mask = (obs_times >= obs_start) & (obs_times <= obs_end)
                observed = observed_full[mask]
                print(f"  Matched observations: {len(observed)} timesteps")
                
                # Resample if needed (obs might be at different frequency)
                if len(observed) != len(times_array):
                    # Simple approach: take first n observations
                    n_sim = len(times_array)
                    if len(observed) >= n_sim:
                        observed = observed[:n_sim]
                    else:
                        # Pad with last value
                        observed = np.pad(observed, (0, n_sim - len(observed)), 
                                         mode='edge')
            else:
                observed = observed_full[:runoff_cms.shape[0]]
            
            print(f"  Observed range: {observed.min():.2f} to {observed.max():.2f} m³/s")
        else:
            print("  No observation file found, using synthetic data")
            _, observed = generate_synthetic_data(runoff_cms.shape[0], n_reaches)
    else:
        print("Using synthetic test data...")
        n_reaches = 10
        n_timesteps = 1000
        runoff_cms, observed = generate_synthetic_data(n_timesteps, n_reaches)
        print(f"  Generated {n_timesteps} timesteps, {n_reaches} reaches")
    
    # Create network if not loaded from topology
    if not HAS_DROUTE:
        print("Cannot run tests without droute")
        return results
    
    if network is None:
        network = create_synthetic_network(n_reaches=runoff_cms.shape[1])
        outlet_idx = runoff_cms.shape[1] - 1
        print(f"\nCreated synthetic network with {network.num_reaches()} reaches")
    else:
        print(f"\nUsing topology network with {network.num_reaches()} reaches")
    
    # Use all available data (already filtered by skip_years/max_years)
    runoff_subset = runoff_cms
    obs_subset = observed[:len(runoff_cms)]  # Match lengths
    times_subset = times_array if times_array is not None else np.arange(len(runoff_cms))
    
    print(f"Testing with {len(runoff_subset)} timesteps")
    print(f"  Outlet reach index: {outlet_idx}")
    
    # Run each method
    print("\n" + "="*60)
    print("Forward Pass Tests")
    print("="*60)
    
    test = RoutingTest(network, runoff_subset, outlet_reach=outlet_idx)
    
    for method in methods:
        try:
            sim = test.run_method(method)
            
            results[f'{method}_forward'] = {
                'nse': nse(sim, obs_subset),
                'kge': kge(sim, obs_subset),
                'rmse': rmse(sim, obs_subset),
                'pbias': pbias(sim, obs_subset),
                'simulated': sim,
            }
            print(f"    NSE={results[f'{method}_forward']['nse']:.3f}, "
                  f"KGE={results[f'{method}_forward']['kge']:.3f}, "
                  f"RMSE={results[f'{method}_forward']['rmse']:.2f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Plot forward pass comparison
    if HAS_MATPLOTLIB:
        print("\n" + "="*60)
        print("Generating Plots")
        print("="*60)
        forward_sims = {method.upper(): results[f'{method}_forward']['simulated'] 
                       for method in methods if f'{method}_forward' in results}
        plot_hydrographs(times_subset, obs_subset, forward_sims,
                        title="Routing Method Comparison (Default Parameters)",
                        filename="routing_comparison_default.png")
    
    # Run optimization if requested
    if optimize:
        print("\n" + "="*60)
        if fast:
            print("Parameter Optimization (Fast Enzyme Kernels)")
        else:
            print("Parameter Optimization (PyTorch + CoDiPack AD)")
        print("="*60)
        
        for method in methods:
            
            try:
                # SVE doesn't support fast Enzyme mode
                if method == 'sve' and fast:
                    print(f"\n  Skipping {method.upper()} - not supported in fast Enzyme mode")
                    continue
                
                # Use the loaded network (or create fresh synthetic one)
                if network is not None:
                    # Reset state on existing network
                    opt_network = network
                else:
                    opt_network = create_synthetic_network(n_reaches=runoff_subset.shape[1])
                
                if fast:
                    # Use fast Enzyme-based optimization
                    start_time = time.time()
                    
                    # Map method name to method index
                    method_map = {
                        'mc': 0,
                        'lag': 1,
                        'irf': 2,
                        'kwt': 3,
                        'diffusive': 4
                    }
                    method_idx = method_map.get(method.lower(), 0)
                    
                    # Use the C++ enzyme optimize function directly with correct method
                    router = droute.enzyme.EnzymeRouter(
                        opt_network, 
                        dt=3600.0, 
                        num_substeps=4,
                        method=method_idx
                    )
                    
                    opt_result = droute.enzyme.optimize(
                        router,
                        runoff_subset.astype(np.float64),
                        obs_subset.astype(np.float64),
                        outlet_idx,
                        n_epochs=30,
                        lr=0.1,
                        eps=0.01,
                        verbose=True
                    )
                    
                    elapsed = time.time() - start_time
                    
                    # Convert lists to numpy arrays if needed (py::cast returns list)
                    sim_data = opt_result['simulated']
                    if isinstance(sim_data, list):
                        sim_data = np.array(sim_data)
                        opt_result['simulated'] = sim_data
                    if isinstance(opt_result.get('optimized_manning_n'), list):
                        opt_result['optimized_manning_n'] = np.array(opt_result['optimized_manning_n'])
                    if isinstance(opt_result.get('losses'), list):
                        opt_result['losses'] = np.array(opt_result['losses'])
                    
                    opt_results = {
                        'nse': opt_result['nse'],
                        'kge': kge(opt_result['simulated'], obs_subset),
                        'rmse': rmse(opt_result['simulated'], obs_subset),
                        'pbias': pbias(opt_result['simulated'], obs_subset),
                        'training_time': elapsed,
                        'final_loss': opt_result['final_loss'],
                        'simulated': opt_result['simulated'],
                        'optimized_manning_n': opt_result['optimized_manning_n'],
                    }
                else:
                    # Use PyTorch + CoDiPack
                    if not HAS_TORCH:
                        print(f"  Skipping {method} - PyTorch not available")
                        continue
                    opt_results = optimize_routing_pytorch(
                        opt_network, runoff_subset, obs_subset,
                        method=method, n_epochs=30, lr=0.05, outlet_reach=outlet_idx
                    )
                
                results[f'{method}_optimized'] = opt_results
                
                print(f"\n  {method.upper()} optimized:")
                print(f"    Final NSE: {opt_results['nse']:.3f}")
                print(f"    Final KGE: {opt_results['kge']:.3f}")
                print(f"    Training time: {opt_results['training_time']:.2f}s")
                
                # Plot before/after comparison
                if HAS_MATPLOTLIB and f'{method}_forward' in results:
                    plot_optimization_comparison(
                        times_subset, obs_subset,
                        results[f'{method}_forward']['simulated'],
                        opt_results['simulated'],
                        method,
                        results[f'{method}_forward'],
                        opt_results,
                        filename=f"optimization_{method}.png"
                    )
                
            except Exception as e:
                print(f"\n  {method.upper()} optimization failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nForward Pass Results:")
    print("-" * 50)
    print(f"{'Method':<12} {'NSE':>8} {'KGE':>8} {'RMSE':>8} {'PBIAS':>8}")
    print("-" * 50)
    for method in methods:
        key = f'{method}_forward'
        if key in results:
            r = results[key]
            print(f"{method.upper():<12} {r['nse']:>8.3f} {r['kge']:>8.3f} "
                  f"{r['rmse']:>8.2f} {r['pbias']:>7.1f}%")
    
    if optimize:
        print("\nOptimization Results:")
        print("-" * 50)
        print(f"{'Method':<12} {'NSE':>8} {'KGE':>8} {'Time':>8}")
        print("-" * 50)
        for method in methods:
            key = f'{method}_optimized'
            if key in results:
                r = results[key]
                print(f"{method.upper():<12} {r['nse']:>8.3f} {r['kge']:>8.3f} "
                      f"{r['training_time']:>7.1f}s")
    
    return results


# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test dRoute routing methods with real data'
    )
    parser.add_argument(
        '--data-dir', type=Path,
        help='Path to data directory'
    )
    parser.add_argument(
        '--optimize', action='store_true',
        help='Run parameter optimization'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Use fast Enzyme kernels for optimization (instead of CoDiPack)'
    )
    parser.add_argument(
        '--skip-years', type=int, default=1,
        help='Years to skip at start for spinup (default: 1)'
    )
    parser.add_argument(
        '--max-years', type=int, default=1,
        help='Maximum years to simulate after spinup (default: 1)'
    )
    parser.add_argument(
        '--methods', nargs='+',
        default=['mc', 'lag', 'irf', 'kwt', 'diffusive'],
        help='Routing methods to test (options: mc, lag, irf, kwt, diffusive, sve)'
    )
    parser.add_argument(
        '--sve', action='store_true',
        help='Include Saint-Venant benchmark (slower, high-fidelity)'
    )
    
    args = parser.parse_args()
    
    # Add SVE to methods if requested
    methods = args.methods.copy()
    if args.sve and 'sve' not in methods:
        methods.append('sve')
    
    # Run tests
    results = run_tests(
        data_dir=args.data_dir,
        optimize=args.optimize,
        fast=args.fast,
        skip_years=args.skip_years,
        max_years=args.max_years,
        methods=methods
    )
    
    return results


if __name__ == '__main__':
    main()
