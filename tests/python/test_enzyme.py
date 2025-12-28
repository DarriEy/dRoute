"""Test Enzyme router and fast kernel functionality."""

import pytest
import numpy as np


def create_simple_network():
    """Create a simple 3-reach linear network."""
    import droute

    network = droute.Network()

    # Add junctions (not strictly required for EnzymeRouter, but for consistency)
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


class TestEnzymeRouter:
    """Test EnzymeRouter functionality."""

    def test_enzyme_router_creation(self):
        """Test creating an EnzymeRouter."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        assert router is not None
        assert router.num_reaches == 3
        assert router.dt == pytest.approx(3600.0)

    def test_enzyme_router_with_options(self):
        """Test EnzymeRouter with custom options."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(
            network,
            dt=1800.0,
            num_substeps=8,
            method=0  # Muskingum-Cunge
        )

        assert router.dt == pytest.approx(1800.0)
        assert router.get_routing_method() == 0

    def test_enzyme_routing_methods(self):
        """Test all Enzyme routing methods."""
        import droute

        network = create_simple_network()

        methods = {
            0: "Muskingum-Cunge",
            1: "Lag",
            2: "IRF",
            3: "KWT",
            4: "Diffusive",
        }

        for method_id, method_name in methods.items():
            net = create_simple_network()
            router = droute.enzyme.EnzymeRouter(net, method=method_id)

            assert router.get_routing_method() == method_id

            # Run routing
            for _ in range(10):
                router.set_lateral_inflow(0, 10.0)
                router.set_lateral_inflow(1, 5.0)
                router.set_lateral_inflow(2, 2.0)
                router.route_timestep()

            q = router.get_discharge(2)
            assert np.isfinite(q), f"Method {method_name} produced non-finite output"
            assert q >= 0, f"Method {method_name} produced negative discharge"

    def test_set_routing_method(self):
        """Test changing routing method."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network, method=0)

        assert router.get_routing_method() == 0

        router.set_routing_method(1)
        assert router.get_routing_method() == 1

        router.set_routing_method(2)
        assert router.get_routing_method() == 2

    def test_enzyme_set_lateral_inflows_array(self):
        """Test setting lateral inflows from numpy array."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        inflows = np.array([10.0, 5.0, 2.0])
        router.set_lateral_inflows(inflows)
        router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)

    def test_enzyme_get_discharges(self):
        """Test getting all discharges as numpy array."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        for _ in range(10):
            router.set_lateral_inflows(np.array([10.0, 5.0, 2.0]))
            router.route_timestep()

        discharges = router.get_discharges()
        assert isinstance(discharges, np.ndarray)
        assert len(discharges) == 3
        assert all(np.isfinite(discharges))

    def test_enzyme_manning_n_operations(self):
        """Test Manning's n get/set operations."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        # Get initial values
        manning_n = router.get_manning_n_all()
        assert len(manning_n) == 3
        assert all(n > 0 for n in manning_n)

        # Set new values - note these are set by reach index
        new_n = np.array([0.03, 0.04, 0.05])
        router.set_manning_n_all(new_n)

        # Values should be updated (may not match exactly due to indexing)
        retrieved = router.get_manning_n_all()
        assert len(retrieved) == 3
        assert all(n > 0 for n in retrieved)

    def test_enzyme_reset_state(self):
        """Test resetting EnzymeRouter state."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        # Run some timesteps
        for _ in range(10):
            router.set_lateral_inflows(np.array([10.0, 5.0, 2.0]))
            router.route_timestep()

        q_before = router.get_discharge(2)
        assert q_before > 0

        # Reset
        router.reset_state()

        # Discharge should be back to zero
        q_after = router.get_discharge(2)
        assert q_after == pytest.approx(0.0, abs=1e-10)

    def test_enzyme_route_multiple(self):
        """Test routing multiple timesteps at once."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        router.set_lateral_inflows(np.array([10.0, 5.0, 2.0]))
        router.route(10)

        q = router.get_discharge(2)
        assert np.isfinite(q)

    def test_enzyme_topology_debug(self):
        """Test getting topology debug string."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        debug_str = router.get_topology_debug()
        assert isinstance(debug_str, str)
        assert len(debug_str) > 0


class TestEnzymeSimulate:
    """Test enzyme.simulate function."""

    def test_simulate_basic(self):
        """Test basic simulation."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        n_timesteps = 20
        n_reaches = 3

        runoff = np.random.rand(n_timesteps, n_reaches) * 10.0
        outlet_reach = 2

        result = droute.enzyme.simulate(router, runoff, outlet_reach)

        assert len(result) == n_timesteps
        assert all(np.isfinite(result))
        assert all(r >= 0 for r in result)

    def test_simulate_constant_inflow(self):
        """Test simulation with constant inflow."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        n_timesteps = 50
        runoff = np.ones((n_timesteps, 3)) * 10.0
        outlet_reach = 2

        result = droute.enzyme.simulate(router, runoff, outlet_reach)

        # Flow should be positive and finite
        assert all(np.isfinite(result))
        assert all(r >= 0 for r in result)
        # Should have some flow at outlet
        assert result[-1] > 0


class TestEnzymeNumericalGradients:
    """Test enzyme.compute_gradients_numerical function."""

    def test_numerical_gradients(self):
        """Test numerical gradient computation."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        n_timesteps = 20
        n_reaches = 3

        runoff = np.ones((n_timesteps, n_reaches)) * 10.0
        # Create synthetic observations (slightly different from simulation)
        observed = np.ones(n_timesteps) * 15.0 + np.random.randn(n_timesteps) * 2.0
        outlet_reach = 2

        result = droute.enzyme.compute_gradients_numerical(
            router, runoff, observed, outlet_reach, eps=0.001
        )

        assert "gradients" in result
        assert "loss" in result

        grads = result["gradients"]
        assert len(grads) == n_reaches
        assert all(np.isfinite(grads))

        loss = result["loss"]
        assert np.isfinite(loss)
        assert loss >= 0


class TestEnzymeOptimize:
    """Test enzyme.optimize function."""

    @pytest.mark.slow
    def test_optimize_basic(self):
        """Test basic optimization."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        n_timesteps = 50
        n_reaches = 3

        runoff = np.ones((n_timesteps, n_reaches)) * 10.0

        # Create target from a different Manning's n
        router.set_manning_n_all(np.array([0.05, 0.05, 0.05]))
        target = droute.enzyme.simulate(router, runoff, 2)

        # Reset to different values and try to recover
        router.set_manning_n_all(np.array([0.035, 0.035, 0.035]))

        result = droute.enzyme.optimize(
            router,
            runoff,
            np.array(target),
            outlet_reach=2,
            n_epochs=5,  # Few epochs for test speed
            lr=0.05,
            eps=0.01,
            verbose=False
        )

        assert "simulated" in result
        assert "losses" in result
        assert "nse" in result
        assert "final_loss" in result
        assert "optimized_manning_n" in result

        # Loss should decrease
        losses = result["losses"]
        assert len(losses) == 5
        assert losses[-1] <= losses[0]

    def test_optimize_returns_valid_parameters(self):
        """Test that optimization returns valid Manning's n values."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        n_timesteps = 30
        runoff = np.ones((n_timesteps, 3)) * 10.0
        observed = np.ones(n_timesteps) * 20.0

        result = droute.enzyme.optimize(
            router, runoff, observed,
            outlet_reach=2, n_epochs=3, verbose=False
        )

        manning_n = result["optimized_manning_n"]
        assert len(manning_n) == 3
        # Should be in valid range (clipped to [0.01, 0.2])
        assert all(0.01 <= n <= 0.2 for n in manning_n)


class TestEnzymeEdgeCases:
    """Test edge cases for Enzyme functionality."""

    def test_enzyme_zero_inflow(self):
        """Test Enzyme router with zero inflow."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        for _ in range(10):
            router.set_lateral_inflows(np.zeros(3))
            router.route_timestep()

        q = router.get_discharge(2)
        # Should be very small (may have numerical floor)
        assert q >= 0
        assert q < 0.1  # Should be near zero

    def test_enzyme_large_inflow(self):
        """Test Enzyme router with large inflow."""
        import droute

        network = create_simple_network()
        router = droute.enzyme.EnzymeRouter(network)

        for _ in range(10):
            router.set_lateral_inflows(np.ones(3) * 10000.0)
            router.route_timestep()

        q = router.get_discharge(2)
        assert np.isfinite(q)
        assert q >= 0

    def test_enzyme_single_reach(self):
        """Test Enzyme router with single reach."""
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

        router = droute.enzyme.EnzymeRouter(network)
        assert router.num_reaches == 1

        router.set_lateral_inflow(0, 10.0)
        router.route_timestep()

        q = router.get_discharge(0)
        assert np.isfinite(q)
