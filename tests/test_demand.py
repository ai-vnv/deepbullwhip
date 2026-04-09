import numpy as np

from deepbullwhip.demand.base import DemandGenerator
from deepbullwhip.demand.semiconductor import SemiconductorDemandGenerator


class TestSemiconductorDemandGenerator:
    def test_deterministic_with_seed(self):
        gen = SemiconductorDemandGenerator()
        d1 = gen.generate(T=100, seed=42)
        d2 = gen.generate(T=100, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_differ(self):
        gen = SemiconductorDemandGenerator()
        d1 = gen.generate(T=100, seed=1)
        d2 = gen.generate(T=100, seed=2)
        assert not np.array_equal(d1, d2)

    def test_shape(self):
        gen = SemiconductorDemandGenerator()
        for T in [10, 50, 156, 300]:
            d = gen.generate(T=T, seed=0)
            assert d.shape == (T,)

    def test_positivity(self):
        gen = SemiconductorDemandGenerator()
        d = gen.generate(T=500, seed=42)
        assert np.all(d >= 0.1)

    def test_shock_raises_mean(self):
        gen = SemiconductorDemandGenerator(shock_period=50, shock_magnitude=0.20)
        d = gen.generate(T=200, seed=42)
        pre_shock = d[10:50].mean()
        post_shock = d[60:200].mean()
        assert post_shock > pre_shock

    def test_no_shock_when_period_exceeds_T(self):
        gen = SemiconductorDemandGenerator(shock_period=1000)
        d = gen.generate(T=100, seed=42)
        gen_no_shock = SemiconductorDemandGenerator(shock_magnitude=0.0)
        d_no = gen_no_shock.generate(T=100, seed=42)
        np.testing.assert_array_equal(d, d_no)

    def test_custom_parameters(self):
        gen = SemiconductorDemandGenerator(mu=100, phi=0.5, sigma_eps=0.05)
        d = gen.generate(T=50, seed=0)
        assert d.shape == (50,)
        assert np.all(d > 0)

    def test_abc_interface(self):
        gen = SemiconductorDemandGenerator()
        assert isinstance(gen, DemandGenerator)
