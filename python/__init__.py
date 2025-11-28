"""
pydmc_route - Differentiable River Routing Library

A high-performance, differentiable routing library for
hydrological modeling and machine learning applications.

Example:
    >>> import pydmc_route as dmc
    >>> 
    >>> # Create network
    >>> network = dmc.Network()
    >>> 
    >>> # Create router
    >>> config = dmc.RouterConfig()
    >>> config.dt = 3600.0
    >>> config.enable_gradients = True
    >>> router = dmc.MuskingumCungeRouter(network, config)
    >>> 
    >>> # Run simulation
    >>> router.start_recording()
    >>> for t in range(num_timesteps):
    ...     network.set_lateral_inflows(inflows[t])
    ...     router.route_timestep()
    >>> router.stop_recording()
    >>> 
    >>> # Compute gradients
    >>> router.compute_gradients([outlet_id], [1.0])
    >>> grads = network.get_grad_manning_n_all()

Available routing methods:
    - MuskingumCungeRouter: Full AD support, 5 learnable parameters
    - IRFRouter: Soft-masked impulse response function
    - DiffusiveWaveRouter: Analytical gradient approximation
    - DiffusiveWaveIFT: Exact gradients via IFT
    - LagRouter: Simple delay (forward-only)
    - KWTRouter: Kinematic wave tracking (forward-only)
    - SoftGatedKWT: Differentiable KWT with soft gates
"""

# The actual module is the compiled C++ extension
# This file serves as documentation and type hints placeholder

__version__ = "0.5.0"
__author__ = "dMC-Route Authors"

# Try to import the compiled extension
try:
    from pydmc_route import *
except ImportError:
    import warnings
    warnings.warn(
        "pydmc_route C++ extension not found. "
        "Build with: cmake -DDMC_BUILD_PYTHON=ON .. && make pydmc_route"
    )
