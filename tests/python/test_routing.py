"""Test routing functionality for all routing methods."""

import pytest
import numpy as np


def create_simple_network():
    """Create a simple 3-reach linear network for testing."""
    import droute

    network = droute.Network()

    # Add junctions first (MuskingumCungeRouter requires explicit junctions)
    for i in range(4):
        junction = droute.Junction()
        junction.id = i
        network.add_junction(junction)

    # Create 3 reaches in a chain: 0 -> 1 -> 2
    for i in range(3):
        reach = droute.Reach()
        reach.id = i
        reach.length = 5000.0  # 5 km
        reach.slope = 0.001
        reach.manning_n = 0.035

        # Set up connectivity via junctions
        reach.upstream_junction_id = i
        reach.downstream_junction_id = i + 1

        network.add_reach(reach)

    network.build_topology()
    return network


def create_branching_network():
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
    reach0.downstream_junction_id = 2  # Junction where 0 and 1 merge
    network.add_reach(reach0)

    # Reach 1: tributary 2
    reach1 = droute.Reach()
    reach1.id = 1
    reach1.length = 4000.0
    reach1.slope = 0.0015
    reach1.manning_n = 0.040
    reach1.upstream_junction_id = 1
    reach1.downstream_junction_id = 2  # Same junction
    network.add_reach(reach1)

    # Reach 2: main stem after confluence
    reach2 = droute.Reach()
    reach2.id = 2
    reach2.length = 10000.0
    reach2.slope = 0.0005
    reach2.manning_n = 0.030
    reach2.upstream_junction_id = 2
    reach2.downstream_junction_id = 3  # Outlet
    network.add_reach(reach2)

    network.build_topology()
    return network


class TestMuskingumCungeRouter:
    """Tests for MuskingumCungeRouter."""

    def test_router_creation(self):
        """Test creating a MuskingumCungeRouter."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        router = droute.MuskingumCungeRouter(network, config)

        assert router is not None
        assert router.current_time() == 0.0

    def test_single_timestep(self):
        """Test routing a single timestep."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dt = 3600.0
        router = droute.MuskingumCungeRouter(network, config)

        # Set lateral inflows
        router.set_lateral_inflow(0, 10.0)
        router.set_lateral_inflow(1, 5.0)
        router.set_lateral_inflow(2, 2.0)

        # Route one timestep
        router.route_timestep()

        # Check outputs are non-negative
        for i in range(3):
            q = router.get_discharge(i)
            assert q >= 0.0

    def test_multiple_timesteps(self):
        """Test routing multiple timesteps."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dt = 3600.0
        router = droute.MuskingumCungeRouter(network, config)

        n_timesteps = 10
        outlet_id = 2

        discharges = []
        for t in range(n_timesteps):
            # Constant inflow
            for i in range(3):
                router.set_lateral_inflow(i, 5.0)
            router.route_timestep()
            discharges.append(router.get_discharge(outlet_id))

        # Flow should build up over time
        assert discharges[-1] > discharges[0]
        # Should approach steady state
        assert all(q >= 0 for q in discharges)

    def test_route_method(self):
        """Test the batch route method."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        router = droute.MuskingumCungeRouter(network, config)

        # Set inflows
        for i in range(3):
            router.set_lateral_inflow(i, 5.0)

        # Route 10 timesteps at once
        router.route(10)

        assert router.current_time() == pytest.approx(10 * 3600.0)

    def test_get_all_discharges(self):
        """Test getting all discharges at once."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        for i in range(3):
            router.set_lateral_inflow(i, 5.0)
        router.route_timestep()

        discharges = router.get_all_discharges()
        assert len(discharges) == 3
        assert all(q >= 0 for q in discharges)

    def test_reset_state(self):
        """Test resetting router state."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        # Run some timesteps
        for _ in range(5):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        initial_q = router.get_discharge(2)
        assert initial_q > 0

        # Reset
        router.reset_state()

        assert router.current_time() == 0.0
        # After reset with no inflow, discharge should be zero
        assert router.get_discharge(2) == pytest.approx(0.0, abs=1e-10)

    def test_set_lateral_inflows_array(self):
        """Test setting lateral inflows from array."""
        import droute

        network = create_simple_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        inflows = [10.0, 5.0, 2.0]
        router.set_lateral_inflows(inflows)
        router.route_timestep()

        # Should produce discharge
        assert router.get_discharge(2) >= 0

    def test_branching_network(self):
        """Test routing on a branching network."""
        import droute

        network = create_branching_network()
        router = droute.MuskingumCungeRouter(network, droute.RouterConfig())

        # Add inflow to tributaries
        router.set_lateral_inflow(0, 10.0)
        router.set_lateral_inflow(1, 8.0)
        router.set_lateral_inflow(2, 0.0)

        # Route several timesteps
        for _ in range(20):
            router.route_timestep()

        # Main stem should have accumulated flow
        q_main = router.get_discharge(2)
        assert q_main > 0


class TestIRFRouter:
    """Tests for IRFRouter (Impulse Response Function)."""

    def test_irf_router_creation(self):
        """Test creating an IRFRouter."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        router = droute.IRFRouter(network, config)

        assert router is not None

    def test_irf_routing(self):
        """Test IRF routing produces output."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.irf_max_kernel_size = 50
        router = droute.IRFRouter(network, config)

        # Add pulse inflow
        router.set_lateral_inflow(0, 100.0)
        router.route_timestep()

        # Clear inflow
        router.set_lateral_inflow(0, 0.0)

        # Route more timesteps - response should decay
        discharges = []
        for _ in range(20):
            router.route_timestep()
            discharges.append(router.get_discharge(2))

        # IRF should produce non-zero outputs
        assert any(q > 0 for q in discharges)


class TestLagRouter:
    """Tests for LagRouter."""

    def test_lag_router_creation(self):
        """Test creating a LagRouter."""
        import droute

        network = create_simple_network()
        router = droute.LagRouter(network, droute.RouterConfig())

        assert router is not None

    def test_lag_routing(self):
        """Test Lag routing with delayed response."""
        import droute

        network = create_simple_network()
        router = droute.LagRouter(network, droute.RouterConfig())

        # Add inflow
        for i in range(3):
            router.set_lateral_inflow(i, 10.0)

        # Route
        for _ in range(10):
            router.route_timestep()

        assert router.get_discharge(2) >= 0


class TestDiffusiveWaveRouter:
    """Tests for DiffusiveWaveRouter."""

    def test_diffusive_wave_creation(self):
        """Test creating a DiffusiveWaveRouter."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dw_num_nodes = 5
        router = droute.DiffusiveWaveRouter(network, config)

        assert router is not None

    def test_diffusive_wave_routing(self):
        """Test diffusive wave routing."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dw_num_nodes = 5
        router = droute.DiffusiveWaveRouter(network, config)

        for i in range(3):
            router.set_lateral_inflow(i, 10.0)

        for _ in range(10):
            router.route_timestep()

        assert router.get_discharge(2) >= 0


class TestKWTRouter:
    """Tests for KWTRouter (Kinematic Wave Tracking)."""

    def test_kwt_router_creation(self):
        """Test creating a KWTRouter."""
        import droute

        network = create_simple_network()
        router = droute.KWTRouter(network, droute.RouterConfig())

        assert router is not None

    def test_kwt_routing(self):
        """Test KWT routing."""
        import droute

        network = create_simple_network()
        router = droute.KWTRouter(network, droute.RouterConfig())

        for i in range(3):
            router.set_lateral_inflow(i, 10.0)

        for _ in range(10):
            router.route_timestep()

        assert router.get_discharge(2) >= 0


class TestDiffusiveWaveIFT:
    """Tests for DiffusiveWaveIFT (IFT adjoint)."""

    def test_dw_ift_creation(self):
        """Test creating DiffusiveWaveIFT router."""
        import droute

        network = create_simple_network()
        router = droute.DiffusiveWaveIFT(network, droute.RouterConfig())

        assert router is not None

    def test_dw_ift_routing(self):
        """Test DiffusiveWaveIFT routing."""
        import droute

        network = create_simple_network()
        router = droute.DiffusiveWaveIFT(network, droute.RouterConfig())

        router.set_lateral_inflow(0, 10.0)
        router.set_lateral_inflow(1, 5.0)
        router.set_lateral_inflow(2, 2.0)

        for _ in range(10):
            router.route_timestep()

        assert router.get_discharge(2) >= 0


class TestSoftGatedKWT:
    """Tests for SoftGatedKWT."""

    def test_soft_kwt_creation(self):
        """Test creating SoftGatedKWT router."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.kwt_gate_steepness = 5.0
        router = droute.SoftGatedKWT(network, config)

        assert router is not None
        assert router.get_steepness() == pytest.approx(5.0)

    def test_soft_kwt_routing(self):
        """Test SoftGatedKWT routing."""
        import droute

        network = create_simple_network()
        router = droute.SoftGatedKWT(network, droute.RouterConfig())

        for i in range(3):
            router.set_lateral_inflow(i, 10.0)

        for _ in range(10):
            router.route_timestep()

        assert router.get_discharge(2) >= 0

    def test_steepness_annealing(self):
        """Test steepness adjustment for annealing."""
        import droute

        network = create_simple_network()
        router = droute.SoftGatedKWT(network, droute.RouterConfig())

        # Change steepness
        router.set_steepness(10.0)
        assert router.get_steepness() == pytest.approx(10.0)

        router.set_steepness(2.0)
        assert router.get_steepness() == pytest.approx(2.0)


class TestRouterConsistency:
    """Test consistency across routing methods."""

    @pytest.mark.slow
    def test_all_routers_produce_output(self):
        """Test that all routers produce non-zero output for same input."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dt = 3600.0

        router_classes = [
            droute.MuskingumCungeRouter,
            droute.IRFRouter,
            droute.LagRouter,
            droute.DiffusiveWaveRouter,
            droute.KWTRouter,
            droute.DiffusiveWaveIFT,
            droute.SoftGatedKWT,
        ]

        results = {}
        for RouterClass in router_classes:
            # Create fresh network for each router
            net = create_simple_network()
            router = RouterClass(net, config)

            # Run simulation
            for _ in range(20):
                for i in range(3):
                    router.set_lateral_inflow(i, 10.0)
                router.route_timestep()

            results[RouterClass.__name__] = router.get_discharge(2)

        # All should produce positive discharge
        for name, q in results.items():
            assert q > 0, f"{name} produced zero discharge"

    def test_mass_conservation_tendency(self):
        """Test that routers produce reasonable output at steady state."""
        import droute

        network = create_simple_network()
        config = droute.RouterConfig()
        config.dt = 3600.0

        router = droute.MuskingumCungeRouter(network, config)

        # Constant total inflow
        total_inflow = 30.0  # 10.0 per reach

        # Run to steady state
        for _ in range(100):
            for i in range(3):
                router.set_lateral_inflow(i, 10.0)
            router.route_timestep()

        # At steady state, outlet discharge should be positive and finite
        outlet_q = router.get_discharge(2)
        assert outlet_q > 0, "Outlet discharge should be positive"
        assert np.isfinite(outlet_q), "Outlet discharge should be finite"
        # Should have some significant flow at outlet
        assert outlet_q > 5.0, "Outlet discharge seems too low"
