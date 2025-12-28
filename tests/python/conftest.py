"""Pytest configuration and fixtures for droute tests."""

import pytest
import sys
from pathlib import Path


def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_data: marks tests that require sample data files"
    )
    config.addinivalue_line(
        "markers", "enzyme: marks tests that require Enzyme AD support"
    )


@pytest.fixture
def simple_network():
    """Create a simple 3-reach linear network for testing."""
    import droute

    network = droute.Network()

    # Add junctions first (required for MuskingumCungeRouter)
    for i in range(4):
        junction = droute.Junction()
        junction.id = i
        network.add_junction(junction)

    for i in range(3):
        reach = droute.Reach()
        reach.id = i
        reach.length = 5000.0  # 5 km
        reach.slope = 0.001
        reach.manning_n = 0.035
        reach.upstream_junction_id = i
        reach.downstream_junction_id = i + 1
        network.add_reach(reach)

    network.build_topology()
    return network


@pytest.fixture
def branching_network():
    """Create a network with tributaries: 0 and 1 merge into 2."""
    import droute

    network = droute.Network()

    # Add junctions first
    for i in range(4):
        junction = droute.Junction()
        junction.id = i
        network.add_junction(junction)

    # Reach 0: tributary 1
    reach0 = droute.Reach()
    reach0.id = 0
    reach0.length = 5000.0
    reach0.slope = 0.001
    reach0.manning_n = 0.035
    reach0.upstream_junction_id = 0
    reach0.downstream_junction_id = 2
    network.add_reach(reach0)

    # Reach 1: tributary 2
    reach1 = droute.Reach()
    reach1.id = 1
    reach1.length = 4000.0
    reach1.slope = 0.0015
    reach1.manning_n = 0.040
    reach1.upstream_junction_id = 1
    reach1.downstream_junction_id = 2
    network.add_reach(reach1)

    # Reach 2: main stem after confluence
    reach2 = droute.Reach()
    reach2.id = 2
    reach2.length = 10000.0
    reach2.slope = 0.0005
    reach2.manning_n = 0.030
    reach2.upstream_junction_id = 2
    reach2.downstream_junction_id = 3
    network.add_reach(reach2)

    network.build_topology()
    return network


@pytest.fixture
def single_reach_network():
    """Create a single-reach network."""
    import droute

    network = droute.Network()

    # Add junctions
    for i in range(2):
        junction = droute.Junction()
        junction.id = i
        network.add_junction(junction)

    reach = droute.Reach()
    reach.id = 0
    reach.length = 5000.0
    reach.slope = 0.001
    reach.manning_n = 0.035
    reach.upstream_junction_id = 0
    reach.downstream_junction_id = 1
    network.add_reach(reach)

    network.build_topology()
    return network


@pytest.fixture
def default_config():
    """Create a default RouterConfig."""
    import droute

    return droute.RouterConfig()


@pytest.fixture
def gradient_config():
    """Create a RouterConfig with gradients enabled."""
    import droute

    config = droute.RouterConfig()
    config.enable_gradients = True
    return config


@pytest.fixture
def sample_runoff():
    """Create sample runoff data for testing."""
    import numpy as np

    n_timesteps = 100
    n_reaches = 3

    # Create realistic runoff pattern with a peak
    t = np.arange(n_timesteps)
    base_flow = 5.0
    peak = 50.0 * np.exp(-((t - 30) ** 2) / 200.0)  # Gaussian peak
    runoff = np.zeros((n_timesteps, n_reaches))

    for i in range(n_reaches):
        # Slight variation between reaches
        runoff[:, i] = base_flow + peak * (1.0 + 0.1 * i)

    return runoff


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Skip slow tests unless explicitly requested
    if not config.getoption("-m"):
        skip_slow = pytest.mark.skip(reason="use -m slow to run slow tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
