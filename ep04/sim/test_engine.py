import numpy as np
import engine as E


def test_expected_value_A_negative_B_positive():
    assert abs(E.expected_value(0.9, 1, 10) - (-0.1)) < 1e-12
    assert abs(E.expected_value(0.3, 3, 1) - 0.2) < 1e-12


def test_kelly_matches_closed_form_B():
    # b=3, p=0.3: (bp-q)/b = (0.9-0.7)/3
    assert abs(E.kelly_fraction(0.3, 3, 1) - 0.2 / 3) < 1e-12
    assert E.kelly_fraction(0.9, 1, 10) == 0.0


def test_kelly_matches_numeric_argmax():
    fs = np.linspace(0, 0.5, 200001)
    g = 0.3 * np.log1p(3 * fs) + 0.7 * np.log1p(-fs)
    assert abs(fs[g.argmax()] - E.kelly_fraction(0.3, 3, 1)) < 1e-4


def test_growth_curve_inverted_U():
    fs = E.kelly_fraction(0.3, 3, 1)
    g = lambda m: E.growth_rate(0.3, 3, 1, m * fs)
    assert g(0.5) > 0 and g(1) > g(0.5) and g(1) > g(1.5)
    assert abs(g(2)) < 0.002          # 两倍凯利附近接近零
    assert g(3) < 0                   # 再多变负


def test_overestimate_p_gives_about_double_kelly():
    assert abs(E.kelly_fraction(0.35, 3, 1) / E.kelly_fraction(0.3, 3, 1) - 2) < 1e-9


def test_zero_position_wealth_unchanged():
    final, logw = E.simulate(0.3, 3, 1, 0.0, 100, 50, seed=1)
    assert np.allclose(final, 1.0) and np.allclose(logw, 0.0)


def test_simulation_growth_matches_theory():
    f = E.kelly_fraction(0.3, 3, 1)
    final, logw = E.simulate(0.3, 3, 1, f, 4000, 1000, seed=7)
    s = E.summarize(final, logw, 1000)
    assert abs(s["growth_per_bet"] - E.growth_rate(0.3, 3, 1, f)) < 5e-4


def test_mean_up_median_down_for_heavy_position():
    f = 3 * E.kelly_fraction(0.3, 3, 1)
    final, logw = E.simulate(0.3, 3, 1, f, 5000, 300, seed=3)
    assert np.median(final) < 1 < final.mean()


def test_seed_reproducible():
    a, _ = E.simulate(0.3, 3, 1, 0.1, 50, 50, seed=5)
    b, _ = E.simulate(0.3, 3, 1, 0.1, 50, 50, seed=5)
    assert np.array_equal(a, b)


def test_vol_target_no_leverage_and_no_lookahead():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    pos = E.vol_target_positions(r, 20, 0.10, 5)
    assert np.nanmax(pos) <= 1.0 and np.isnan(pos[:20]).all()
    r2 = r.copy(); r2[300] = 0.5          # 改动第300天不能影响第300天及以前的仓位
    pos2 = E.vol_target_positions(r2, 20, 0.10, 5)
    assert np.array_equal(pos[:301], pos2[:301], equal_nan=True)


def test_shuffle_false_positive_rate_about_5pct():
    """无波动聚集的独立收益：择时没有信号，显著率应约 5%。"""
    rng = np.random.default_rng(42)
    hits, trials = 0, 300
    for k in range(trials):
        r = rng.normal(0.0003, 0.01, 750)
        pos = E.vol_target_positions(r, 20, 0.12, 5)
        ok = ~np.isnan(pos)
        obs = E.sharpe(E.apply_positions(r, pos))
        null = E.shuffle_benchmark(r, pos, 199, seed=k)
        hits += E.one_sided_p(obs, null) <= 0.05
    assert 0.02 <= hits / trials <= 0.09, hits / trials


def test_shuffle_detects_real_vol_clustering_signal():
    """有波动聚集且均值为零时，调仓应降低回撤；至少最大回撤更小。"""
    rng = np.random.default_rng(1)
    n = 3000
    vol = np.empty(n); vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.exp(0.97 * np.log(vol[t-1]) + 0.03 * np.log(0.01) + rng.normal(0, 0.15))
    r = rng.normal(0.0004, 1, n) * vol
    pos = E.vol_target_positions(r, 20, 0.12, 5)
    strat = E.apply_positions(r, pos)
    hold = r[~np.isnan(pos)]
    assert E.max_drawdown(strat) > E.max_drawdown(hold)
