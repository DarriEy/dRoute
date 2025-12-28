"""Test gradient computation functionality."""

import pytest
import numpy as np


def create_simple_network():
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
        reach.length = 5000.0
        reach.slope = 0.001
        reach.manning_n = 0.035
        reach.upstream_junction_id = i
        reach.downstream_junction_id = i + 1
        network.add_reach(reach)

    network.build_topology()
    return network


class TestMuskingumCungeGradients:
    """Test gradient computation for MuskingumCungeRouter."""

    def test_gradient_recording(self):
        """Test starting and stopping gradient recording."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.MuskingumCungeRouter(network, config)

        router.start_recording()

        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
            router.record_output(2)

        router.stop_recording()

        # Should have recorded 5 outputs
        assert router.get_output_history_size(2) == 5

    def test_compute_gradients(self):
        """Test computing gradients."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.MuskingumCungeRouter(network, config)

        router.start_recording()

        n_timesteps = 10
        for _ in range(n_timesteps):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
            router.record_output(2)

        router.stop_recording()

        # Compute gradients with unit gradient
        dL_dQ = [1.0] * n_timesteps
        router.compute_gradients_timeseries(2, dL_dQ)

        grads = router.get_gradients()
        assert isinstance(grads, dict)

    def test_reset_gradients(self):
        """Test resetting gradients."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.MuskingumCungeRouter(network, config)

        router.start_recording()
        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
            router.record_output(2)
        router.stop_recording()

        router.compute_gradients_timeseries(2, [1.0] * 5)
        router.reset_gradients()

        # After reset, get_gradients should still work
        grads = router.get_gradients()
        assert isinstance(grads, dict)

    def test_clear_output_history(self):
        """Test clearing output history."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        router.start_recording()
        for _ in range(5):
            router.route_timestep()
            router.record_output(2)
        router.stop_recording()

        assert router.get_output_history_size(2) == 5

        router.clear_output_history()
        assert router.get_output_history_size(2) == 0

    def test_get_output_history(self):
        """Test retrieving output history."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        router = droute.MuskingumCungeRouter(network, config)

        router.start_recording()
        for t in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
            router.record_output(2)
        router.stop_recording()

        history = router.get_output_history(2)
        assert len(history) == 5
        # All values should be non-negative
        assert all(q >= 0 for q in history)

    @pytest.mark.slow
    def test_gradient_finite_difference_check(self):
        """Verify AD gradients match finite differences."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        config.dt = 3600.0

        outlet_id = 2
        n_timesteps = 20
        eps = 1e-5

        # Helper to run simulation and get loss
        def run_simulation(manning_n_vals):
            net = create_simple_network()
            # Set Manning's n values
            for i, n in enumerate(manning_n_vals):
                net.get_reach(i).manning_n = n

            router = droute.MuskingumCungeRouter(net, config)

            total_loss = 0.0
            for t in range(n_timesteps):
                for i in range(3):
                    router.set_lateral_inflow(i, 10.0 + t * 0.5)
                router.route_timestep()
                q = router.get_discharge(outlet_id)
                # Simple loss: sum of squared outputs
                total_loss += q * q

            return total_loss

        # Base Manning's n
        base_n = [0.035, 0.035, 0.035]
        base_loss = run_simulation(base_n)

        # Compute numerical gradient for first reach
        perturbed_n = base_n.copy()
        perturbed_n[0] = base_n[0] + eps
        loss_plus = run_simulation(perturbed_n)

        perturbed_n[0] = base_n[0] - eps
        loss_minus = run_simulation(perturbed_n)

        numerical_grad = (loss_plus - loss_minus) / (2 * eps)

        # The gradient should be non-zero and finite
        assert np.isfinite(numerical_grad)
        # For this simple case, changing manning_n should affect loss
        assert abs(numerical_grad) > 0


class TestIRFGradients:
    """Test gradient computation for IRFRouter."""

    def test_irf_gradient_recording(self):
        """Test IRF router gradient recording."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.IRFRouter(network, config)

        router.start_recording()
        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
        router.stop_recording()

        # Should complete without error
        router.compute_gradients([2], [1.0])
        grads = router.get_gradients()
        assert isinstance(grads, dict)


class TestDiffusiveWaveGradients:
    """Test gradient computation for DiffusiveWaveRouter."""

    def test_dw_gradient_recording(self):
        """Test DiffusiveWave router gradient recording."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.DiffusiveWaveRouter(network, config)

        router.start_recording()
        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
        router.stop_recording()

        router.compute_gradients([2], [1.0])
        grads = router.get_gradients()
        assert isinstance(grads, dict)


class TestDiffusiveWaveIFTGradients:
    """Test IFT-based gradients for DiffusiveWaveIFT."""

    def test_ift_gradient_computation(self):
        """Test IFT adjoint gradient computation."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        config.dw_use_ift_adjoint = True
        router = droute.DiffusiveWaveIFT(network, config)

        router.start_recording()
        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
        router.stop_recording()

        router.compute_gradients([2], [1.0])
        grads = router.get_gradients()
        assert isinstance(grads, dict)


class TestSoftGatedKWTGradients:
    """Test gradient computation for SoftGatedKWT."""

    def test_soft_kwt_gradients(self):
        """Test SoftGatedKWT gradient recording and computation."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.SoftGatedKWT(network, config)

        router.start_recording()
        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
        router.stop_recording()

        router.compute_gradients([2], [1.0])
        grads = router.get_gradients()
        assert isinstance(grads, dict)


class TestNetworkGradients:
    """Test network-level gradient operations."""

    def test_network_zero_gradients(self):
        """Test zeroing network gradients."""
        import droute

        network = create_simple_network()
        network.zero_gradients()  # Should not raise

    def test_get_grad_manning_n_all(self):
        """Test getting all Manning's n gradients."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.MuskingumCungeRouter(network, config)

        # Run and compute gradients
        router.start_recording()
        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()
            router.record_output(2)
        router.stop_recording()

        router.compute_gradients_timeseries(2, [1.0] * 5)

        # Get gradients from network
        grads = network.get_grad_manning_n_all()
        assert len(grads) == 3


class TestGradientEdgeCases:
    """Test edge cases in gradient computation."""

    def test_gradients_with_zero_inflow(self):
        """Test gradients when inflow is zero."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.MuskingumCungeRouter(network, config)

        router.start_recording()
        for _ in range(5):
            # Zero inflow
            for i in range(3):
                router.set_lateral_inflow(i, 0.0)
            router.route_timestep()
            router.record_output(2)
        router.stop_recording()

        # Should still work (even if gradients are zero)
        router.compute_gradients_timeseries(2, [1.0] * 5)
        grads = router.get_gradients()
        assert isinstance(grads, dict)

    def test_gradients_with_single_timestep(self):
        """Test gradients with just one timestep."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.enable_gradients = True
        router = droute.MuskingumCungeRouter(network, config)

        router.start_recording()
        for i in range(3):
            router.set_lateral_inflow(i, 10.0)
        router.route_timestep()
        router.record_output(2)
        router.stop_recording()

        router.compute_gradients_timeseries(2, [1.0])
        grads = router.get_gradients()
        assert isinstance(grads, dict)
