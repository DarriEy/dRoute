"""Test RouterConfig functionality."""

import pytest


class TestRouterConfig:
    """Tests for RouterConfig class."""

    def test_default_config(self):
        """Test default RouterConfig values."""
        import droute

        config = droute.RouterConfig()

        # Basic settings
        assert config.dt == 3600.0
        assert config.enable_gradients is True
        assert config.min_flow == pytest.approx(1e-6)

        # Muskingum bounds
        assert config.x_lower_bound == pytest.approx(0.0)
        assert config.x_upper_bound == pytest.approx(0.5)

    def test_config_modification(self):
        """Test modifying RouterConfig values."""
        import droute

        config = droute.RouterConfig()

        # Modify settings
        config.dt = 1800.0
        config.enable_gradients = False
        config.min_flow = 1e-8
        config.num_substeps = 8

        assert config.dt == 1800.0
        assert config.enable_gradients is False
        assert config.min_flow == pytest.approx(1e-8)
        assert config.num_substeps == 8

    def test_config_smooth_bounds(self):
        """Test smooth bounds settings."""
        import droute

        config = droute.RouterConfig()

        # Default smooth bounds
        assert config.use_smooth_bounds is True
        assert config.smooth_epsilon == pytest.approx(1e-6)

        # Modify
        config.use_smooth_bounds = False
        config.smooth_epsilon = 1e-4
        assert config.use_smooth_bounds is False
        assert config.smooth_epsilon == pytest.approx(1e-4)

    def test_config_substepping(self):
        """Test substepping configuration."""
        import droute

        config = droute.RouterConfig()

        # Default substepping
        assert config.fixed_substepping is True
        assert config.num_substeps == 4
        assert config.adaptive_substepping is False
        assert config.max_substeps == 10

        # Test adaptive mode
        config.fixed_substepping = False
        config.adaptive_substepping = True
        config.max_substeps = 20

        assert config.fixed_substepping is False
        assert config.adaptive_substepping is True
        assert config.max_substeps == 20

    def test_config_irf_options(self):
        """Test IRF-specific configuration."""
        import droute

        config = droute.RouterConfig()

        # Default values (verify they exist and are reasonable)
        assert config.irf_max_kernel_size > 0
        assert config.irf_shape_param > 0
        assert isinstance(config.irf_soft_mask, bool)
        assert config.irf_mask_steepness > 0

    def test_config_diffusive_wave_options(self):
        """Test diffusive wave configuration."""
        import droute

        config = droute.RouterConfig()

        assert config.dw_num_nodes == 10
        assert config.dw_use_ift_adjoint is True

    def test_config_kwt_options(self):
        """Test KWT-specific configuration."""
        import droute

        config = droute.RouterConfig()

        assert config.kwt_gate_steepness == pytest.approx(5.0)
        assert config.kwt_anneal_steepness is False
        assert config.kwt_steepness_min == pytest.approx(1.0)
        assert config.kwt_steepness_max == pytest.approx(20.0)

    def test_config_checkpointing(self):
        """Test checkpointing configuration."""
        import droute

        config = droute.RouterConfig()

        assert config.enable_checkpointing is False
        assert config.checkpoint_interval == 1000

        config.enable_checkpointing = True
        config.checkpoint_interval = 500

        assert config.enable_checkpointing is True
        assert config.checkpoint_interval == 500

    def test_config_parallel(self):
        """Test parallel routing configuration."""
        import droute

        config = droute.RouterConfig()

        assert config.parallel_routing is False
        assert config.num_threads == 4

    def test_config_repr(self):
        """Test RouterConfig string representation."""
        import droute

        config = droute.RouterConfig()
        repr_str = repr(config)

        assert "RouterConfig" in repr_str
        assert "dt=" in repr_str
        assert "enable_gradients=" in repr_str
