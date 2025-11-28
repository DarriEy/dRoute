#!/usr/bin/env python3
"""
dMC-Route Calibration using PyTorch Autograd

Uses dMC-Route's Jacobian output combined with PyTorch for gradient-based
parameter optimization. The Jacobian (dQ/dθ) from dMC-Route is used in a
custom autograd function to enable end-to-end differentiation.

IMPORTANT NOTES:
1. Routing parameters (Manning's n) affect TIMING, not total volume
2. If your model underestimates magnitude, that's a runoff generation issue
3. This script optimizes for hydrograph shape/timing, not magnitude

Usage:
    python calibrate_routing.py --config config.txt --obs observations.csv
    python calibrate_routing.py -c config.txt --obs obs.csv --epochs 50 --lr 0.01
"""

import argparse
import subprocess
import numpy as np
import pandas as pd
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Install with: pip install torch")


# ============================================================================
# Working Directory Management
# ============================================================================

class WorkingDirectory:
    """Manages a temporary working directory for calibration."""
    
    def __init__(self, original_config: str, routing_method: str = None):
        self.original_config = original_config
        self.routing_method = routing_method
        self.work_dir = None
        self.work_config = None
        self.work_topology = None
        self.work_output = None
        self.work_jacobian = None
        self.work_manning_n = None  # NEW: CSV file for Manning's n
        
    def setup(self) -> 'WorkingDirectory':
        """Create working directory with copies of necessary files."""
        self.work_dir = Path(tempfile.mkdtemp(prefix='dmc_calibration_'))
        
        # Parse original config
        config_lines = []
        topology_path = None
        
        with open(self.original_config, 'r') as f:
            config_lines = f.readlines()
        
        for line in config_lines:
            if '<fname_ntopo>' in line or '<network_file>' in line:
                topology_path = self._extract_value(line)
        
        if not topology_path:
            raise ValueError("Could not find topology path (<fname_ntopo>) in config")
        
        # Just reference original topology (no need to copy since we use -m flag)
        self.work_topology = topology_path
        
        # Set output paths in working directory
        self.work_output = self.work_dir / 'discharge.csv'
        self.work_jacobian = self.work_dir / 'jacobian.csv'
        self.work_manning_n = self.work_dir / 'manning_n.csv'
        
        # Create modified config pointing to work files
        self.work_config = self.work_dir / 'config.txt'
        with open(self.work_config, 'w') as f:
            for line in config_lines:
                if '<fname_output>' in line:
                    f.write(f"<fname_output>  {self.work_output}\n")
                elif '<output_file>' in line:
                    f.write(f"<output_file>  {self.work_output}\n")
                elif '<fname_jacobian>' in line:
                    f.write(f"<fname_jacobian>  {self.work_jacobian}\n")
                elif '<jacobian_file>' in line:
                    f.write(f"<jacobian_file>  {self.work_jacobian}\n")
                else:
                    f.write(line)
        
        print(f"Working directory: {self.work_dir}")
        return self
    
    def _extract_value(self, line: str) -> Optional[str]:
        """Extract value from <tag> value format."""
        if '>' in line:
            rest = line.split('>')[1]
            if '!' in rest:
                rest = rest.split('!')[0]
            return rest.strip()
        return None
    
    def cleanup(self):
        """Remove working directory."""
        if self.work_dir and self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)


# ============================================================================
# Model Runner
# ============================================================================

def write_manning_n_csv(filepath: str, reach_ids: np.ndarray, 
                        manning_n: np.ndarray) -> bool:
    """Write Manning's n values to a CSV file."""
    try:
        with open(filepath, 'w') as f:
            f.write("reach_id,manning_n\n")
            for rid, n_val in zip(reach_ids, manning_n):
                f.write(f"{int(rid)},{float(n_val)}\n")
        return True
    except Exception as e:
        print(f"Warning: Failed to write Manning's n CSV: {e}")
        return False


def run_dmc_route(exe: str, config: str, manning_n_file: str = None, 
                  routing_method: str = None, verbose: bool = False) -> bool:
    """Run dMC-Route and return success status."""
    cmd = [exe, '-c', str(config)]
    if manning_n_file:
        cmd.extend(['-m', str(manning_n_file)])
    if routing_method:
        cmd.extend(['-r', routing_method])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if verbose:
        print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def load_results(discharge_csv: str, jacobian_csv: str, 
                 sim_start: str = '1990-01-01') -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load model outputs with proper datetime conversion."""
    Q_df = pd.read_csv(discharge_csv)
    jac_df = pd.read_csv(jacobian_csv)
    
    # Get outlet (last Q column)
    q_cols = [c for c in Q_df.columns if c.startswith('Q_')]
    Q_sim = Q_df[q_cols[-1]].values
    times = Q_df['time'].values
    
    # Convert times to datetime (times are seconds since sim_start)
    start_dt = pd.to_datetime(sim_start)
    datetimes = [start_dt + timedelta(seconds=float(t)) for t in times]
    Q_df['datetime'] = datetimes
    
    return Q_df, Q_sim, jac_df


# ============================================================================
# Time Series Alignment
# ============================================================================

def align_timeseries(sim_df: pd.DataFrame, obs_df: pd.DataFrame, 
                     sim_col: str, obs_col: str,
                     spinup_years: int = 1) -> Tuple[np.ndarray, np.ndarray, datetime, datetime]:
    """Align simulation and observation timeseries."""
    if 'datetime' not in sim_df.columns:
        raise ValueError("Simulation must have 'datetime' column")
    
    sim_df = sim_df.copy()
    obs_df = obs_df.copy()
    
    sim_df['datetime'] = pd.to_datetime(sim_df['datetime'])
    
    dt_col = None
    for col in ['datetime', 'date', 'time', 'DATE', 'DateTime']:
        if col in obs_df.columns:
            dt_col = col
            break
    if dt_col is None:
        dt_col = obs_df.columns[0]
    
    obs_df['datetime'] = pd.to_datetime(obs_df[dt_col])
    
    sim_start = sim_df['datetime'].min()
    spinup_end = sim_start + pd.DateOffset(years=spinup_years)
    sim_df = sim_df[sim_df['datetime'] >= spinup_end]
    
    print(f"Simulation period: {sim_df['datetime'].min()} to {sim_df['datetime'].max()}")
    print(f"Observation period: {obs_df['datetime'].min()} to {obs_df['datetime'].max()}")
    print(f"Spinup skipped: {spinup_years} year(s) (data before {spinup_end})")
    
    overlap_start = max(sim_df['datetime'].min(), obs_df['datetime'].min())
    overlap_end = min(sim_df['datetime'].max(), obs_df['datetime'].max())
    
    if overlap_start >= overlap_end:
        raise ValueError(f"No overlapping period!")
    
    print(f"Overlapping period: {overlap_start} to {overlap_end}")
    
    sim_overlap = sim_df[(sim_df['datetime'] >= overlap_start) & 
                         (sim_df['datetime'] <= overlap_end)].set_index('datetime')
    obs_overlap = obs_df[(obs_df['datetime'] >= overlap_start) & 
                         (obs_df['datetime'] <= overlap_end)].set_index('datetime')
    
    merged = sim_overlap[[sim_col]].join(obs_overlap[[obs_col]], how='inner')
    print(f"Matched timesteps: {len(merged)}")
    
    return merged[sim_col].values, merged[obs_col].values, overlap_start, overlap_end


# ============================================================================
# Metrics
# ============================================================================

def compute_nse(Q_sim: np.ndarray, Q_obs: np.ndarray) -> float:
    """Compute Nash-Sutcliffe Efficiency."""
    mask = ~(np.isnan(Q_sim) | np.isnan(Q_obs))
    Q_s, Q_o = Q_sim[mask], Q_obs[mask]
    if len(Q_s) == 0:
        return -999
    return 1 - np.sum((Q_s - Q_o)**2) / np.sum((Q_o - np.mean(Q_o))**2)


def compute_kge(Q_sim: np.ndarray, Q_obs: np.ndarray) -> float:
    """Compute Kling-Gupta Efficiency."""
    mask = ~(np.isnan(Q_sim) | np.isnan(Q_obs))
    Q_s, Q_o = Q_sim[mask], Q_obs[mask]
    if len(Q_s) == 0:
        return -999
    r = np.corrcoef(Q_s, Q_o)[0, 1]
    alpha = np.std(Q_s) / np.std(Q_o)
    beta = np.mean(Q_s) / np.mean(Q_o)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)


def compute_nse_torch(Q_sim: torch.Tensor, Q_obs: torch.Tensor) -> torch.Tensor:
    """Compute NSE using PyTorch (differentiable)."""
    mask = ~(torch.isnan(Q_sim) | torch.isnan(Q_obs))
    Q_s = Q_sim[mask]
    Q_o = Q_obs[mask]
    ss_res = torch.sum((Q_s - Q_o)**2)
    ss_tot = torch.sum((Q_o - torch.mean(Q_o))**2)
    return 1 - ss_res / (ss_tot + 1e-10)


def compute_kge_torch(Q_sim: torch.Tensor, Q_obs: torch.Tensor) -> torch.Tensor:
    """Compute KGE using PyTorch (differentiable)."""
    mask = ~(torch.isnan(Q_sim) | torch.isnan(Q_obs))
    Q_s = Q_sim[mask]
    Q_o = Q_obs[mask]
    
    Q_s_centered = Q_s - torch.mean(Q_s)
    Q_o_centered = Q_o - torch.mean(Q_o)
    r = torch.sum(Q_s_centered * Q_o_centered) / (
        torch.sqrt(torch.sum(Q_s_centered**2)) * torch.sqrt(torch.sum(Q_o_centered**2)) + 1e-10
    )
    
    alpha = torch.std(Q_s) / (torch.std(Q_o) + 1e-10)
    beta = torch.mean(Q_s) / (torch.mean(Q_o) + 1e-10)
    
    return 1 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


# ============================================================================
# PyTorch Custom Function for dMC-Route
# ============================================================================

class DMCRouteFunction(torch.autograd.Function):
    """Custom autograd function that wraps dMC-Route."""
    
    @staticmethod
    def forward(ctx, manning_n: torch.Tensor, 
                exe: str, work_dir: WorkingDirectory,
                sim_start: str, obs_df: pd.DataFrame, obs_col: str, 
                spinup_years: int, reach_ids: np.ndarray) -> torch.Tensor:
        """Forward pass: update parameters, run dMC-Route, return aligned Q."""
        
        # Write Manning's n to CSV file
        manning_n_np = manning_n.detach().numpy()
        write_manning_n_csv(work_dir.work_manning_n, reach_ids, manning_n_np)
        
        # Run model with working config and manning_n file
        if not run_dmc_route(exe, work_dir.work_config, 
                             manning_n_file=str(work_dir.work_manning_n),
                             routing_method=work_dir.routing_method,
                             verbose=False):
            raise RuntimeError("dMC-Route failed")
        
        # Load results from working directory
        sim_df, _, jac_df = load_results(
            str(work_dir.work_output), 
            str(work_dir.work_jacobian), 
            sim_start
        )
        
        # Get outlet column
        q_cols = [c for c in sim_df.columns if c.startswith('Q_')]
        sim_col = q_cols[-1]
        
        # Align timeseries (silently)
        sim_df_copy = sim_df.copy()
        obs_df_copy = obs_df.copy()
        
        sim_df_copy['datetime'] = pd.to_datetime(sim_df_copy['datetime'])
        dt_col = 'datetime' if 'datetime' in obs_df_copy.columns else obs_df_copy.columns[0]
        obs_df_copy['datetime'] = pd.to_datetime(obs_df_copy[dt_col])
        
        sim_start_dt = sim_df_copy['datetime'].min()
        spinup_end = sim_start_dt + pd.DateOffset(years=spinup_years)
        sim_df_copy = sim_df_copy[sim_df_copy['datetime'] >= spinup_end]
        
        overlap_start = max(sim_df_copy['datetime'].min(), obs_df_copy['datetime'].min())
        overlap_end = min(sim_df_copy['datetime'].max(), obs_df_copy['datetime'].max())
        
        sim_overlap = sim_df_copy[(sim_df_copy['datetime'] >= overlap_start) & 
                                   (sim_df_copy['datetime'] <= overlap_end)].set_index('datetime')
        obs_overlap = obs_df_copy[(obs_df_copy['datetime'] >= overlap_start) & 
                                   (obs_df_copy['datetime'] <= overlap_end)].set_index('datetime')
        
        merged = sim_overlap[[sim_col]].join(obs_overlap[[obs_col]], how='inner')
        
        Q_sim = torch.tensor(merged[sim_col].values, dtype=torch.float32)
        
        # Save Jacobian for backward pass
        ctx.jacobian = jac_df
        ctx.n_reaches = len(jac_df)
        ctx.save_for_backward(manning_n)
        
        return Q_sim
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: compute gradient using Jacobian."""
        manning_n, = ctx.saved_tensors
        jac_df = ctx.jacobian
        
        # Sum gradient over time
        dL_dQ_sum = grad_output.sum().item()
        
        # Get dQ/dn from Jacobian
        dQ_dn = jac_df['manning_n'].values
        
        # Chain rule: dL/dn = dL/dQ * dQ/dn
        grad_manning_n = torch.tensor(dL_dQ_sum * dQ_dn, dtype=torch.float32)
        
        return grad_manning_n, None, None, None, None, None, None, None


# ============================================================================
# PyTorch Optimization Loop
# ============================================================================

def optimize_with_torch(exe: str, work_dir: WorkingDirectory,
                        obs_df: pd.DataFrame, obs_col: str, sim_start: str,
                        spinup_years: int, n_reaches: int, reach_ids: np.ndarray,
                        n_epochs: int = 50, lr: float = 0.01,
                        loss_type: str = 'nse') -> Tuple[Dict, np.ndarray]:
    """Optimize Manning's n using PyTorch."""
    
    # Initialize Manning's n
    manning_n = torch.nn.Parameter(
        torch.full((n_reaches,), 0.03, dtype=torch.float32),
        requires_grad=True
    )
    
    optimizer = optim.Adam([manning_n], lr=lr)
    
    dt_col = 'datetime' if 'datetime' in obs_df.columns else obs_df.columns[0]
    obs_df['datetime'] = pd.to_datetime(obs_df[dt_col])
    
    history = {
        'epoch': [], 'loss': [], 'nse': [], 'kge': [],
        'manning_n_mean': [], 'grad_norm': []
    }
    
    print(f"\n{'='*60}")
    print(f"PyTorch Optimization")
    print(f"{'='*60}")
    print(f"Epochs: {n_epochs}, Learning Rate: {lr}, Loss: {loss_type}")
    print(f"{'='*60}\n")
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        try:
            Q_sim = DMCRouteFunction.apply(
                manning_n, exe, work_dir,
                sim_start, obs_df.copy(), obs_col, spinup_years, reach_ids
            )
        except RuntimeError as e:
            print(f"Epoch {epoch}: Model failed - {e}")
            break
        
        # Load aligned obs
        sim_df, _, _ = load_results(
            str(work_dir.work_output), 
            str(work_dir.work_jacobian), 
            sim_start
        )
        q_cols = [c for c in sim_df.columns if c.startswith('Q_')]
        sim_col = q_cols[-1]
        
        sim_df_copy = sim_df.copy()
        obs_df_copy = obs_df.copy()
        sim_df_copy['datetime'] = pd.to_datetime(sim_df_copy['datetime'])
        
        sim_start_dt = sim_df_copy['datetime'].min()
        spinup_end = sim_start_dt + pd.DateOffset(years=spinup_years)
        sim_df_copy = sim_df_copy[sim_df_copy['datetime'] >= spinup_end]
        
        overlap_start = max(sim_df_copy['datetime'].min(), obs_df_copy['datetime'].min())
        overlap_end = min(sim_df_copy['datetime'].max(), obs_df_copy['datetime'].max())
        
        sim_overlap = sim_df_copy[(sim_df_copy['datetime'] >= overlap_start) & 
                                   (sim_df_copy['datetime'] <= overlap_end)].set_index('datetime')
        obs_overlap = obs_df_copy[(obs_df_copy['datetime'] >= overlap_start) & 
                                   (obs_df_copy['datetime'] <= overlap_end)].set_index('datetime')
        
        merged = sim_overlap[[sim_col]].join(obs_overlap[[obs_col]], how='inner')
        Q_obs = torch.tensor(merged[obs_col].values, dtype=torch.float32)
        
        # Compute loss
        if loss_type == 'nse':
            metric = compute_nse_torch(Q_sim, Q_obs)
            loss = -metric
        elif loss_type == 'kge':
            metric = compute_kge_torch(Q_sim, Q_obs)
            loss = -metric
        else:
            mask = ~(torch.isnan(Q_sim) | torch.isnan(Q_obs))
            loss = torch.mean((Q_sim[mask] - Q_obs[mask])**2)
            metric = -loss
        
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([manning_n], max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            manning_n.clamp_(0.01, 0.15)
        
        # Record
        nse_val = compute_nse(Q_sim.detach().numpy(), Q_obs.numpy())
        kge_val = compute_kge(Q_sim.detach().numpy(), Q_obs.numpy())
        
        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['nse'].append(nse_val)
        history['kge'].append(kge_val)
        history['manning_n_mean'].append(manning_n.mean().item())
        history['grad_norm'].append(float(grad_norm))
        
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, NSE={nse_val:.4f}, "
                  f"KGE={kge_val:.4f}, mean(n)={manning_n.mean().item():.4f}")
    
    return history, manning_n.detach().numpy()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calibrate dMC-Route parameters using PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', '-c', required=True, help='Control file')
    parser.add_argument('--obs', required=True, help='Observations CSV')
    parser.add_argument('--obs-col', default='discharge_cms', help='Observation column')
    parser.add_argument('--exe', default='./dmc_route_run', help='dMC-Route executable')
    parser.add_argument('--routing', '-r', default=None, 
                        choices=['muskingum', 'irf', 'diffusive', 'lag', 'kwt'],
                        help='Routing method (kwt not differentiable)')
    parser.add_argument('--sim-start', default='1990-01-01', help='Time reference date')
    parser.add_argument('--spinup', type=int, default=1, help='Spinup years')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--loss', choices=['nse', 'kge', 'mse'], default='nse')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("dMC-Route Parameter Calibration")
    print("=" * 60)
    
    # Setup working directory
    work_dir = WorkingDirectory(args.config, routing_method=args.routing)
    
    try:
        work_dir.setup()
        
        # Load observations
        obs_df = pd.read_csv(args.obs)
        print(f"\nObservation file: {args.obs}")
        print(f"Observation columns: {list(obs_df.columns)}")
        
        obs_col = args.obs_col
        if obs_col not in obs_df.columns:
            for alt in ['Q', 'discharge', 'streamflow', 'flow', 'discharge_cms']:
                if alt in obs_df.columns:
                    obs_col = alt
                    break
            else:
                print(f"Error: Cannot find discharge column")
                return
        
        print(f"Using: {obs_col}")
        if args.routing:
            print(f"Routing method: {args.routing}")
        
        # Initial run
        print(f"\nRunning dMC-Route...")
        if not run_dmc_route(args.exe, work_dir.work_config, routing_method=args.routing):
            print("Model run failed!")
            return
        
        sim_df, Q_sim_raw, jac_df = load_results(
            str(work_dir.work_output), 
            str(work_dir.work_jacobian), 
            args.sim_start
        )
        print(f"Timesteps: {len(Q_sim_raw)}, Reaches: {len(jac_df)}")
        
        q_cols = [c for c in sim_df.columns if c.startswith('Q_')]
        sim_col = q_cols[-1]
        
        print(f"\n--- Time Alignment ---")
        try:
            Q_sim, Q_obs, start_date, end_date = align_timeseries(
                sim_df, obs_df.copy(), sim_col, obs_col, args.spinup
            )
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # Initial metrics
        nse = compute_nse(Q_sim, Q_obs)
        kge = compute_kge(Q_sim, Q_obs)
        bias = np.nanmean(Q_sim) / np.nanmean(Q_obs)
        
        print(f"\n=== Initial Performance ===")
        print(f"NSE: {nse:.4f}, KGE: {kge:.4f}, Bias: {bias:.4f}")
        print(f"Mean Obs: {np.nanmean(Q_obs):.2f}, Mean Sim: {np.nanmean(Q_sim):.2f}")
        
        # Jacobian info
        print(f"\n=== Jacobian ===")
        jac_df['abs_grad'] = np.abs(jac_df['manning_n'])
        for _, row in jac_df.nlargest(5, 'abs_grad').iterrows():
            print(f"  Reach {int(row['reach_id'])}: dQ/dn = {row['manning_n']:.6f}")
        
        if args.eval_only:
            print("\n[--eval-only specified]")
            return
        
        if not HAS_TORCH:
            print("\nError: PyTorch required")
            return
        
        reach_ids = jac_df['reach_id'].values
        
        history, final_n = optimize_with_torch(
            exe=args.exe,
            work_dir=work_dir,
            obs_df=obs_df,
            obs_col=obs_col,
            sim_start=args.sim_start,
            spinup_years=args.spinup,
            n_reaches=len(jac_df),
            reach_ids=reach_ids,
            n_epochs=args.epochs,
            lr=args.lr,
            loss_type=args.loss
        )
        
        print(f"\n{'='*60}")
        print("Optimization Complete")
        print(f"{'='*60}")
        print(f"NSE: {history['nse'][0]:.4f} → {history['nse'][-1]:.4f}")
        print(f"KGE: {history['kge'][0]:.4f} → {history['kge'][-1]:.4f}")
        print(f"Mean n: {final_n.mean():.4f}")
        
        # Save
        pd.DataFrame(history).to_csv('optimization_history.csv', index=False)
        pd.DataFrame({'reach_id': reach_ids, 'manning_n': final_n}).to_csv(
            'optimized_parameters.csv', index=False
        )
        print("\nSaved: optimization_history.csv, optimized_parameters.csv")
        
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                
                axes[0,0].plot(history['nse'])
                axes[0,0].set_ylabel('NSE')
                axes[0,0].set_title('NSE')
                axes[0,0].grid(True)
                
                axes[0,1].plot(history['kge'])
                axes[0,1].set_ylabel('KGE')
                axes[0,1].set_title('KGE')
                axes[0,1].grid(True)
                
                axes[1,0].plot(history['manning_n_mean'])
                axes[1,0].set_ylabel("Mean n")
                axes[1,0].set_title("Manning's n")
                axes[1,0].grid(True)
                
                axes[1,1].semilogy(history['grad_norm'])
                axes[1,1].set_ylabel('||grad||')
                axes[1,1].set_title('Gradient')
                axes[1,1].grid(True)
                
                plt.tight_layout()
                plt.savefig('optimization_results.png', dpi=150)
                print("Saved: optimization_results.png")
            except ImportError:
                pass
        
    finally:
        work_dir.cleanup()
        print(f"\nCleaned up working directory")


if __name__ == '__main__':
    main()
