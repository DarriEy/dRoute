#!/usr/bin/env python3
"""
Example: Differentiable Parameter Learning with dMC-Route

This example shows how to use dMC-Route's gradient capabilities
for parameter optimization. The approach couples the routing model
with a gradient-based optimizer (Adam) to learn Manning's n from
observed streamflow.

This is a conceptual example - the actual implementation would use
the compiled library via ctypes or pybind11 bindings.
"""

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Placeholder for the actual library binding
# In practice, you would load the compiled library:
# import ctypes
# lib = ctypes.CDLL("libdmc_route.so")

class DMCRouteModel:
    """
    Python wrapper for the dMC-Route BMI model.
    
    This is a placeholder showing the expected interface.
    In production, this would call the actual C++ library.
    """
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.n_reaches = 3  # Example: 3-reach network
        self.dt = 3600.0    # 1 hour
        
        # Parameters (learnable)
        self.manning_n = np.full(self.n_reaches, 0.035)
        
        # State
        self.Q = np.zeros(self.n_reaches)
        self.t = 0.0
        
    def initialize(self):
        """Initialize the model."""
        pass
        
    def set_parameters(self, name: str, values: np.ndarray):
        """Set parameter values."""
        if name == "manning_n":
            self.manning_n = values.copy()
            
    def get_parameters(self, name: str) -> np.ndarray:
        """Get parameter values."""
        if name == "manning_n":
            return self.manning_n.copy()
        return np.array([])
        
    def set_lateral_inflow(self, inflows: np.ndarray):
        """Set lateral inflow for all reaches."""
        self.lateral_inflow = inflows.copy()
        
    def forward(self, lateral_inflows: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Run forward simulation and return outlet discharge time series.
        
        In the actual implementation, this would:
        1. Call StartRecording() to start AD tape
        2. Loop: SetValue("lateral_inflow"), Update()
        3. Call StopRecording()
        4. Return discharge at outlet
        """
        Q_series = np.zeros(n_steps)
        
        # Simplified Muskingum routing (placeholder)
        for t in range(n_steps):
            # This is a very simplified approximation
            # The real model uses proper MC equations with AD
            inflow = lateral_inflows[t]
            self.Q[-1] = 0.8 * self.Q[-1] + 0.2 * inflow * (0.04 / self.manning_n[-1])
            Q_series[t] = self.Q[-1]
            
        return Q_series
        
    def compute_gradients(self, dL_dQ: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradients via reverse AD.
        
        In the actual implementation:
        1. SetOutputGradients("discharge", dL_dQ)
        2. ComputeGradients()
        3. GetParameterGradients("manning_n", grad_n)
        
        For Muskingum-Cunge: higher n = more friction = slower flow = lower peak Q
        So dQ/dn < 0 (increasing n decreases Q)
        """
        # Placeholder: compute approximate gradient
        # dQ/dn is negative (more friction = less flow)
        # The actual dMC-Route computes this correctly via AD
        dQ_dn = -self.Q[-1] / self.manning_n[-1]  # Approximate: Q proportional to 1/n
        grad_n = dL_dQ[-1] * dQ_dn
        
        return {
            "manning_n": np.array([grad_n] * self.n_reaches)
        }
        
    def reset(self):
        """Reset model state."""
        self.Q = np.zeros(self.n_reaches)
        self.t = 0.0


class AdamOptimizer:
    """Adam optimizer for parameter learning."""
    
    def __init__(self, params: Dict[str, np.ndarray], lr: float = 0.001,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.params = {k: v.copy() for k, v in params.items()}
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        
    def step(self, grads: Dict[str, np.ndarray]):
        """Update parameters using gradients."""
        self.t += 1
        
        for name, grad in grads.items():
            if name not in self.params:
                continue
                
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            self.params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # Clamp Manning's n to valid range
            if name == "manning_n":
                self.params[name] = np.clip(self.params[name], 0.01, 0.15)


def generate_synthetic_observations(n_true: float, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic observations using 'true' Manning's n.
    
    Returns forcing data and observed discharge.
    """
    # Simple storm hydrograph
    forcing = np.zeros(n_steps)
    peak = n_steps // 4
    for t in range(n_steps):
        if t < peak:
            forcing[t] = 10 * t / peak
        elif t < peak * 2:
            forcing[t] = 10 * (1 - (t - peak) / peak)
        else:
            forcing[t] = 0.0
            
    # Generate observations with true model
    model = DMCRouteModel("config.yaml")
    model.manning_n = np.full(model.n_reaches, n_true)
    Q_obs = model.forward(forcing, n_steps)
    
    # Add observation noise
    Q_obs += np.random.normal(0, 0.1, n_steps)
    Q_obs = np.maximum(Q_obs, 0)
    
    return forcing, Q_obs


def mse_loss(Q_sim: np.ndarray, Q_obs: np.ndarray) -> float:
    """Mean squared error loss."""
    return np.mean((Q_sim - Q_obs) ** 2)


def mse_gradient(Q_sim: np.ndarray, Q_obs: np.ndarray) -> np.ndarray:
    """Gradient of MSE w.r.t. Q_sim."""
    return 2 * (Q_sim - Q_obs) / len(Q_sim)


def train(model: DMCRouteModel, 
          forcing: np.ndarray, 
          Q_obs: np.ndarray,
          n_epochs: int = 100,
          lr: float = 0.001) -> Dict[str, List[float]]:
    """
    Train Manning's n using gradient descent.
    
    Args:
        model: dMC-Route model instance
        forcing: Lateral inflow time series
        Q_obs: Observed discharge time series
        n_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Training history
    """
    # Initialize optimizer
    optimizer = AdamOptimizer(
        params={"manning_n": model.get_parameters("manning_n")},
        lr=lr
    )
    
    history = {
        "loss": [],
        "manning_n": [],
        "nse": []
    }
    
    n_steps = len(forcing)
    
    for epoch in range(n_epochs):
        # Reset model state
        model.reset()
        
        # Set current parameters
        model.set_parameters("manning_n", optimizer.params["manning_n"])
        
        # Forward pass
        Q_sim = model.forward(forcing, n_steps)
        
        # Compute loss
        loss = mse_loss(Q_sim, Q_obs)
        
        # Compute NSE
        nse = 1 - np.sum((Q_sim - Q_obs) ** 2) / np.sum((Q_obs - np.mean(Q_obs)) ** 2)
        
        # Compute gradients
        dL_dQ = mse_gradient(Q_sim, Q_obs)
        grads = model.compute_gradients(dL_dQ)
        
        # Update parameters
        optimizer.step(grads)
        
        # Record history
        history["loss"].append(loss)
        history["manning_n"].append(optimizer.params["manning_n"][-1])
        history["nse"].append(nse)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss:.4f}, n = {optimizer.params['manning_n'][-1]:.4f}, NSE = {nse:.3f}")
    
    # Update model with final parameters
    model.set_parameters("manning_n", optimizer.params["manning_n"])
    
    return history


def plot_results(history: Dict[str, List[float]], 
                 forcing: np.ndarray, 
                 Q_obs: np.ndarray,
                 Q_final: np.ndarray,
                 n_true: float):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss curve
    ax = axes[0, 0]
    ax.semilogy(history["loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.grid(True)
    
    # Parameter convergence
    ax = axes[0, 1]
    ax.plot(history["manning_n"], label="Learned n")
    ax.axhline(n_true, color='r', linestyle='--', label=f"True n = {n_true}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Manning's n")
    ax.set_title("Parameter Convergence")
    ax.legend()
    ax.grid(True)
    
    # Hydrograph comparison
    ax = axes[1, 0]
    t = np.arange(len(Q_obs))
    ax.plot(t, Q_obs, 'k-', label="Observed", linewidth=2)
    ax.plot(t, Q_final, 'b--', label="Simulated", linewidth=1.5)
    ax.fill_between(t, 0, forcing / forcing.max() * Q_obs.max() * 0.3, 
                    alpha=0.3, label="Forcing (scaled)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title("Hydrograph Comparison")
    ax.legend()
    ax.grid(True)
    
    # NSE curve
    ax = axes[1, 1]
    ax.plot(history["nse"])
    ax.axhline(0, color='gray', linestyle=':')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NSE")
    ax.set_title("Nash-Sutcliffe Efficiency")
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150)
    print("\nResults saved to training_results.png")


def main():
    print("=" * 60)
    print("dMC-Route: Differentiable Parameter Learning Example")
    print("=" * 60)
    print()
    
    # True parameter value (unknown to the optimizer)
    n_true = 0.03
    
    # Generate synthetic data
    print("Generating synthetic observations...")
    n_steps = 100
    forcing, Q_obs = generate_synthetic_observations(n_true, n_steps)
    
    # Initialize model with wrong parameter
    print("Initializing model with n = 0.05 (true = 0.03)...")
    model = DMCRouteModel("config.yaml")
    model.manning_n = np.full(model.n_reaches, 0.05)  # Start far from truth
    
    # Train
    print("\nTraining...\n")
    history = train(model, forcing, Q_obs, n_epochs=50, lr=0.005)
    
    # Final simulation
    model.reset()
    Q_final = model.forward(forcing, n_steps)
    
    # Report
    print("\n" + "=" * 60)
    print("Results:")
    print(f"  True Manning's n:    {n_true:.4f}")
    print(f"  Learned Manning's n: {model.manning_n[-1]:.4f}")
    print(f"  Final NSE:           {history['nse'][-1]:.3f}")
    print("=" * 60)
    
    # Plot
    try:
        plot_results(history, forcing, Q_obs, Q_final, n_true)
    except Exception as e:
        print(f"Could not create plot: {e}")


if __name__ == "__main__":
    main()
