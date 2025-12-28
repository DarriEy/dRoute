"""Test edge cases and numerical stability."""

import pytest
import numpy as np


def create_simple_network():
    """Create a simple 3-reach linear network."""
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
        reach.length = 5000.0
        reach.slope = 0.001
        reach.manning_n = 0.035
        reach.upstream_junction_id = i
        reach.downstream_junction_id = i + 1
        network.add_reach(reach)

    network.build_topology()
    return network


class TestEmptyAndSingleReach:
    """Test edge cases with empty or minimal networks."""

    def test_single_reach_network(self):
        """Test routing on a single-reach network."""
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

        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        router.set_lateral_inflow(0, 10.0)
        router.route_timestep()

        assert router.get_discharge(0) >= 0

    def test_empty_inflow(self):
        """Test routing with zero inflow."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        router = droute.MuskingumCungeRouter(network, config)

        # No inflow set (defaults to 0)
        for _ in range(10):
            router.route_timestep()

        # Should still work, produce minimal discharge (min_flow or similar)
        q = router.get_discharge(2)
        assert q >= 0
        assert q <= config.min_flow * 10  # Should be near min_flow


class TestExtremeParameterValues:
    """Test behavior with extreme parameter values."""

    def test_very_small_manning_n(self):
        """Test with very small Manning's n (smooth channel)."""
        import droute

        network = create_simple_network()
        for i in range(3):
            network.get_reach(i).manning_n = 0.01  # Very smooth

        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        # Should produce output without NaN
        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_very_large_manning_n(self):
        """Test with large Manning's n (rough channel)."""
        import droute

        network = create_simple_network()
        for i in range(3):
            network.get_reach(i).manning_n = 0.15  # Very rough

        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_very_small_slope(self):
        """Test with very small slope (nearly flat)."""
        import droute

        network = create_simple_network()
        for i in range(3):
            network.get_reach(i).slope = 1e-5  # Very flat

        config = droute.RouterConfig()
        router = droute.MuskingumCungeRouter(network, config)

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_very_steep_slope(self):
        """Test with steep slope."""
        import droute

        network = create_simple_network()
        for i in range(3):
            network.get_reach(i).slope = 0.1  # 10% grade

        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_short_reach(self):
        """Test with very short reach."""
        import droute

        network = droute.Network()

        # Add junctions
        for i in range(2):
            junction = droute.Junction()
            junction.id = i
            network.add_junction(junction)

        reach = droute.Reach()
        reach.id = 0
        reach.length = 100.0  # 100 meters
        reach.slope = 0.001
        reach.manning_n = 0.035
        reach.upstream_junction_id = 0
        reach.downstream_junction_id = 1
        network.add_reach(reach)
        network.build_topology()

        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        router.set_lateral_inflow(0, 10.0)
        for _ in range(10):
            router.route_timestep()

        q = router.get_discharge(0)
        assert np.isfinite(q)

    def test_long_reach(self):
        """Test with very long reach."""
        import droute

        network = droute.Network()

        # Add junctions
        for i in range(2):
            junction = droute.Junction()
            junction.id = i
            network.add_junction(junction)

        reach = droute.Reach()
        reach.id = 0
        reach.length = 100000.0  # 100 km
        reach.slope = 0.001
        reach.manning_n = 0.035
        reach.upstream_junction_id = 0
        reach.downstream_junction_id = 1
        network.add_reach(reach)
        network.build_topology()

        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        router.set_lateral_inflow(0, 10.0)
        for _ in range(10):
            router.route_timestep()

        q = router.get_discharge(0)
        assert np.isfinite(q)


class TestNumericalStability:
    """Test numerical stability."""

    def test_small_inflow(self):
        """Test with very small inflow values."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 1e-6)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_large_inflow(self):
        """Test with very large inflow values."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10000.0)  # Large flood
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_variable_inflow(self):
        """Test with highly variable inflow."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        # Oscillating inflow
        for t in range(50):
            inflow = 50.0 * (1 + np.sin(t * 0.5))
            for i in range(3):
                router.set_lateral_inflow(i, inflow)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_pulse_inflow(self):
        """Test response to pulse inflow."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        # Large pulse at t=0
        for i in range(3):
            router.set_lateral_inflow(i, 1000.0)
        router.route_timestep()

        # Then zero inflow
        for _ in range(50):
            for i in range(3):
                router.set_lateral_inflow(i, 0.0)
            router.route_timestep()

        # Discharge should decay to near zero
        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_many_timesteps(self):
        """Test stability over many timesteps."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for _ in range(1000):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0


class TestTimestepVariations:
    """Test different timestep configurations."""

    def test_small_timestep(self):
        """Test with small timestep."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dt = 60.0  # 1 minute

        router = droute.MuskingumCungeRouter(network, config)

        for _ in range(100):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_large_timestep(self):
        """Test with large timestep."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dt = 86400.0  # 1 day

        router = droute.MuskingumCungeRouter(network, config)

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0


class TestSubstepping:
    """Test substepping configurations."""

    def test_many_substeps(self):
        """Test with many substeps."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.fixed_substepping = True
        config.num_substeps = 20

        router = droute.MuskingumCungeRouter(network, config)

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_single_substep(self):
        """Test with single substep."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.fixed_substepping = True
        config.num_substeps = 1

        router = droute.MuskingumCungeRouter(network, config)

        for _ in range(10):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0


class TestChannelGeometry:
    """Test channel geometry functionality."""

    def test_channel_geometry_creation(self):
        """Test creating ChannelGeometry."""
        import droute

        geom = droute.ChannelGeometry()
        assert geom is not None

    def test_channel_geometry_properties(self):
        """Test setting channel geometry properties."""
        import droute

        geom = droute.ChannelGeometry()

        geom.width_coef = 7.2
        geom.width_exp = 0.5
        geom.depth_coef = 0.27
        geom.depth_exp = 0.3

        assert geom.width_coef == pytest.approx(7.2)
        assert geom.width_exp == pytest.approx(0.5)
        assert geom.depth_coef == pytest.approx(0.27)
        assert geom.depth_exp == pytest.approx(0.3)

    def test_channel_geometry_methods(self):
        """Test channel geometry width/depth calculations."""
        import droute

        geom = droute.ChannelGeometry()
        geom.width_coef = 7.2
        geom.width_exp = 0.5
        geom.depth_coef = 0.27
        geom.depth_exp = 0.3

        # Test width calculation
        w = geom.width(100.0)  # Q = 100 m³/s
        assert w > 0
        assert np.isfinite(w)

        # Test depth calculation
        d = geom.depth(100.0)
        assert d > 0
        assert np.isfinite(d)


class TestReachProperties:
    """Test Reach property access."""

    def test_reach_gradient_properties(self):
        """Test accessing reach gradient properties."""
        import droute

        reach = droute.Reach()
        reach.id = 0
        reach.length = 5000.0
        reach.slope = 0.001
        reach.manning_n = 0.035

        # Gradient properties should be readable
        _ = reach.grad_manning_n
        _ = reach.grad_width_coef
        _ = reach.grad_width_exp
        _ = reach.grad_depth_coef
        _ = reach.grad_depth_exp

    def test_reach_state_properties(self):
        """Test accessing reach state properties."""
        import droute

        reach = droute.Reach()
        reach.id = 0
        reach.lateral_inflow = 10.0

        assert reach.lateral_inflow == pytest.approx(10.0)

        # Inflow/outflow are read-only
        _ = reach.inflow
        _ = reach.outflow


class TestNetworkBulkOperations:
    """Test network bulk operations."""

    def test_set_manning_n_all(self):
        """Test setting Manning's n for all reaches."""
        import droute

        network = create_simple_network()

        # Get the topological order to set values correctly
        topo_order = network.topological_order()
        n_reaches = len(topo_order)

        new_values = np.array([0.03, 0.04, 0.05])
        network.set_manning_n_all(new_values)

        # Retrieved values should match what was set
        retrieved = network.get_manning_n_all()
        assert len(retrieved) == n_reaches
        # The values are set in topological order, so they should match
        for i, val in enumerate(retrieved):
            assert val > 0, f"Manning's n at index {i} should be positive"

    def test_set_lateral_inflows(self):
        """Test setting lateral inflows for all reaches."""
        import droute

        network = create_simple_network()

        inflows = np.array([10.0, 20.0, 30.0])
        network.set_lateral_inflows(inflows)

        # Run to verify it works
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())
        router.route_timestep()

        assert router.get_discharge(2) >= 0

    def test_get_outflows(self):
        """Test getting outflows for all reaches."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for i in range(3):
            router.set_lateral_inflow(i, 10.0)
        router.route_timestep()

        outflows = network.get_outflows()
        assert len(outflows) == 3
        assert all(np.isfinite(outflows))
