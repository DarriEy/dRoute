"""Test network topology functionality."""

import pytest


def test_network_creation():
    """Test creating an empty network."""
    import droute

    network = droute.Network()
    assert network is not None


def test_reach_creation():
    """Test creating a reach."""
    import droute

    reach = droute.Reach()
    assert reach is not None

    # Test setting basic properties
    reach.id = 0
    reach.length = 5000.0
    reach.slope = 0.001
    reach.manning_n = 0.035
    reach.downstream_junction_id = -1

    assert reach.id == 0
    assert reach.length == 5000.0
    assert reach.downstream_junction_id == -1


def test_network_add_reach():
    """Test adding reaches to a network."""
    import droute

    network = droute.Network()

    for i in range(5):
        reach = droute.Reach()
        reach.id = i
        reach.length = 5000.0
        reach.slope = 0.001
        reach.manning_n = 0.035
        reach.downstream_junction_id = i + 1 if i < 4 else -1
        network.add_reach(reach)

    # Should have 5 reaches
    assert network.num_reaches() == 5
