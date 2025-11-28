#!/usr/bin/env python3
"""
PyTorch Integration Example for dMC-Route

This example demonstrates how to use dMC-Route with PyTorch for
differentiable hydrological modeling. The C++ gradients are
transferred directly to PyTorch tensors without disk I/O.

Requirements:
    pip install torch numpy

Build Python bindings:
    cd build
    cmake .. -DDMC_BUILD_PYTHON=ON
    make pydmc_route
    
Usage:
    # Add build/python to PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python
    python examples/pytorch_routing.py
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Install with: pip install torch")

try:
    import pydmc_route as dmc
    HAS_DMC = True
except ImportError:
    HAS_DMC = False
    print("pydmc_route not found. Build with: cmake -DDMC_BUILD_PYTHON=ON .. && make pydmc_route")


class DMCRoutingFunction(torch.autograd.Function):
    """
    PyTorch autograd function wrapping dMC-Route.
    
    Forward: Run routing simulation, return outlet discharge
    Backward: Use dMC-Route's AD gradients, transfer to PyTorch
    """
    
    @staticmethod
    def forward(ctx, manning_n, lateral_inflows, router, network, 
                gauge_reaches, num_timesteps):
        """
        Forward pass: run routing simulation.
        
        Args:
            manning_n: torch.Tensor of shape (num_reaches,)
            lateral_inflows: torch.Tensor of shape (num_timesteps, num_reaches)
            router: dMC router instance
            network: dMC network instance
            gauge_reaches: list of gauge reach IDs
            num_timesteps: number of timesteps to simulate
            
        Returns:
            outlet_discharge: torch.Tensor of shape (num_timesteps,)
        """
        # Store for backward
        ctx.router = router
        ctx.network = network
        ctx.gauge_reaches = gauge_reaches
        ctx.num_reaches = manning_n.shape[0]
        
        # Convert to numpy
        n_values = manning_n.detach().cpu().numpy()
        inflows = lateral_inflows.detach().cpu().numpy()
        
        # Set Manning's n values
        network.set_manning_n_all(n_values)
        
        # Reset and prepare for simulation
        router.reset_state()
        router.reset_gradients()
        router.start_recording()
        
        # Run simulation
        outlet_discharge = []
        for t in range(num_timesteps):
            network.set_lateral_inflows(inflows[t, :])
            router.route_timestep()
            
            # Get outlet discharge (first gauge reach)
            Q_out = router.get_discharge(gauge_reaches[0])
            outlet_discharge.append(Q_out)
        
        router.stop_recording()
        
        # Convert to tensor
        return torch.tensor(outlet_discharge, dtype=manning_n.dtype)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients w.r.t. manning_n.
        
        Args:
            grad_output: gradient of loss w.r.t. outlet_discharge
            
        Returns:
            Gradients for (manning_n, lateral_inflows, router, network, 
                          gauge_reaches, num_timesteps)
        """
        router = ctx.router
        network = ctx.network
        gauge_reaches = ctx.gauge_reaches
        
        # Compute gradients using dMC-Route's AD
        # Sum of gradients weighted by grad_output
        dL_dQ = grad_output.sum().item()  # Simple case: scalar loss
        
        router.compute_gradients(gauge_reaches, [dL_dQ])
        
        # Get gradients for all reaches
        grad_n = network.get_grad_manning_n_all()
        grad_manning_n = torch.tensor(grad_n, dtype=grad_output.dtype)
        
        # Return gradients (None for non-tensor inputs)
        return grad_manning_n, None, None, None, None, None


class DMCRoutingLayer(nn.Module):
    """
    PyTorch module for differentiable river routing.
    
    Wraps dMC-Route as a learnable layer that can be integrated
    into neural network architectures.
    """
    
    def __init__(self, num_reaches, reach_ids, gauge_reaches,
                 dt=3600.0, method='muskingum'):
        """
        Initialize routing layer.
        
        Args:
            num_reaches: number of reaches in network
            reach_ids: list of reach IDs in topological order
            gauge_reaches: list of gauge reach IDs for loss computation
            dt: timestep in seconds
            method: routing method ('muskingum', 'irf', 'diffusive', etc.)
        """
        super().__init__()
        
        self.num_reaches = num_reaches
        self.reach_ids = reach_ids
        self.gauge_reaches = gauge_reaches
        
        # Create network
        self.network = dmc.Network()
        self._build_simple_chain_network()
        
        # Create router config
        config = dmc.RouterConfig()
        config.dt = dt
        config.enable_gradients = True
        config.use_smooth_bounds = True
        
        # Create router based on method
        if method == 'muskingum':
            self.router = dmc.MuskingumCungeRouter(self.network, config)
        elif method == 'irf':
            self.router = dmc.IRFRouter(self.network, config)
        elif method == 'diffusive':
            self.router = dmc.DiffusiveWaveRouter(self.network, config)
        elif method == 'diffusive-ift':
            self.router = dmc.DiffusiveWaveIFT(self.network, config)
        elif method == 'kwt-soft':
            self.router = dmc.SoftGatedKWT(self.network, config)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Learnable parameters: Manning's n for each reach
        # Initialize with typical values
        self.manning_n = nn.Parameter(
            torch.full((num_reaches,), 0.035, dtype=torch.float64)
        )
    
    def _build_simple_chain_network(self):
        """Build a simple chain network for testing."""
        for i, reach_id in enumerate(self.reach_ids):
            reach = dmc.Reach()
            reach.id = reach_id
            reach.name = f"Reach_{reach_id}"
            reach.length = 5000.0  # 5 km
            reach.slope = 0.001
            reach.manning_n = 0.035
            reach.geometry.width_coef = 10.0
            reach.geometry.width_exp = 0.5
            reach.geometry.depth_coef = 0.4
            reach.geometry.depth_exp = 0.3
            
            # Chain connectivity
            if i > 0:
                reach.upstream_junction_id = reach_id - 1
            if i < len(self.reach_ids) - 1:
                reach.downstream_junction_id = reach_id
            
            self.network.add_reach(reach)
        
        # Add junctions
        for i in range(len(self.reach_ids) - 1):
            junc = dmc.Junction()
            junc.id = self.reach_ids[i]
            junc.upstream_reach_ids = [self.reach_ids[i]]
            junc.downstream_reach_ids = [self.reach_ids[i + 1]]
            self.network.add_junction(junc)
        
        self.network.build_topology()
    
    def forward(self, lateral_inflows):
        """
        Forward pass: route lateral inflows through network.
        
        Args:
            lateral_inflows: tensor of shape (num_timesteps, num_reaches)
            
        Returns:
            outlet_discharge: tensor of shape (num_timesteps,)
        """
        num_timesteps = lateral_inflows.shape[0]
        
        return DMCRoutingFunction.apply(
            self.manning_n,
            lateral_inflows,
            self.router,
            self.network,
            self.gauge_reaches,
            num_timesteps
        )


def example_calibration():
    """
    Example: Calibrate Manning's n to match observed discharge.
    """
    print("=" * 60)
    print("dMC-Route + PyTorch Calibration Example")
    print("=" * 60)
    
    # Configuration
    num_reaches = 5
    num_timesteps = 100
    reach_ids = list(range(1, num_reaches + 1))
    gauge_reaches = [num_reaches]  # Outlet
    
    # Create routing layer
    routing = DMCRoutingLayer(
        num_reaches=num_reaches,
        reach_ids=reach_ids,
        gauge_reaches=gauge_reaches,
        dt=3600.0,
        method='muskingum'
    )
    
    # Generate synthetic lateral inflows (sinusoidal pattern)
    t = np.arange(num_timesteps)
    base_inflow = 10.0 * (1 + 0.5 * np.sin(2 * np.pi * t / 24))
    lateral_inflows = torch.tensor(
        np.column_stack([base_inflow * (1 + 0.1 * i) for i in range(num_reaches)]),
        dtype=torch.float64
    )
    
    # Generate "observed" discharge using true parameters
    true_n = torch.full((num_reaches,), 0.030, dtype=torch.float64)
    with torch.no_grad():
        routing.manning_n.copy_(true_n)
        observed = routing(lateral_inflows)
    
    # Reset to initial guess
    routing.manning_n.data.fill_(0.050)
    
    # Optimizer
    optimizer = torch.optim.Adam([routing.manning_n], lr=0.001)
    
    print("\nCalibrating Manning's n...")
    print(f"True values:    {true_n.numpy()}")
    print(f"Initial guess:  {routing.manning_n.data.numpy()}")
    
    # Training loop
    for epoch in range(50):
        optimizer.zero_grad()
        
        # Forward pass
        predicted = routing(lateral_inflows)
        
        # Loss: MSE
        loss = torch.mean((predicted - observed) ** 2)
        
        # Backward pass
        loss.backward()
        
        # Update
        optimizer.step()
        
        # Clamp to valid range
        with torch.no_grad():
            routing.manning_n.clamp_(0.01, 0.1)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.item():.6f}, "
                  f"n = {routing.manning_n.data.numpy()}")
    
    print(f"\nFinal values:   {routing.manning_n.data.numpy()}")
    print(f"True values:    {true_n.numpy()}")
    print(f"Final loss:     {loss.item():.6f}")


def example_simple_forward():
    """
    Example: Simple forward pass without PyTorch.
    """
    print("=" * 60)
    print("dMC-Route Simple Forward Pass Example")
    print("=" * 60)
    
    # Create network
    network = dmc.Network()
    
    # Add reaches
    for i in range(1, 4):
        reach = dmc.Reach()
        reach.id = i
        reach.name = f"Reach_{i}"
        reach.length = 5000.0
        reach.slope = 0.001
        reach.manning_n = 0.035
        reach.geometry.width_coef = 10.0
        reach.geometry.width_exp = 0.5
        reach.geometry.depth_coef = 0.4
        reach.geometry.depth_exp = 0.3
        
        if i > 1:
            reach.upstream_junction_id = i - 1
        if i < 3:
            reach.downstream_junction_id = i
        
        network.add_reach(reach)
    
    # Add junctions
    for i in range(1, 3):
        junc = dmc.Junction()
        junc.id = i
        junc.upstream_reach_ids = [i]
        junc.downstream_reach_ids = [i + 1]
        network.add_junction(junc)
    
    network.build_topology()
    
    # Create router
    config = dmc.RouterConfig()
    config.dt = 3600.0
    config.enable_gradients = True
    
    router = dmc.MuskingumCungeRouter(network, config)
    
    # Set lateral inflows
    inflows = np.array([10.0, 5.0, 2.0])
    network.set_lateral_inflows(inflows)
    
    # Run simulation
    print("\nRunning 24-hour simulation...")
    router.start_recording()
    
    for t in range(24):
        router.route_timestep()
        Q_out = router.get_discharge(3)
        print(f"  Hour {t+1:2d}: Q_outlet = {Q_out:.2f} m³/s")
    
    router.stop_recording()
    
    # Compute gradients
    router.compute_gradients([3], [1.0])
    
    # Get gradients
    grads = network.get_grad_manning_n_all()
    print(f"\nGradients (∂Q_outlet/∂n): {grads}")


if __name__ == "__main__":
    if not HAS_DMC:
        print("\n" + "=" * 60)
        print("ERROR: pydmc_route module not found!")
        print("=" * 60)
        print("\nTo build Python bindings:")
        print("  cd build")
        print("  cmake .. -DDMC_BUILD_PYTHON=ON")
        print("  make pydmc_route")
        print("  export PYTHONPATH=$PYTHONPATH:$(pwd)/python")
        exit(1)
    
    # Run examples
    example_simple_forward()
    
    if HAS_TORCH:
        print("\n")
        example_calibration()
    else:
        print("\nSkipping PyTorch example (torch not installed)")
