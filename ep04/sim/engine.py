"""第四期计算引擎：期望值、凯利、多人模拟、按波动调仓、洗牌基准。

只含纯函数，不读任何真实数据。真实数据由 run_all.py 在预注册提交后调用。
"""
import numpy as np


def expected_value(p, win, loss):
    """单次期望：p*win - (1-p)*loss（loss 为正数）。"""
    return p * win - (1 - p) * loss


def kelly_fraction(p, win, loss):
    """解析凯利值：最大化 p*ln(1+f*win) + (1-p)*ln(1-f*loss)。
    f* = p/loss - (1-p)/win，负值截为 0。"""
    return max(0.0, p / loss - (1 - p) / win)


def growth_rate(p, win, loss, f):
    """每次下注的期望对数增长率；仓位使输时资金归零则为 -inf。"""
    if f * loss >= 1:
        return float("-inf")
    return p * np.log1p(f * win) + (1 - p) * np.log1p(-f * loss)


def simulate(p, win, loss, f, n_people, n_bets, seed):
    """n_people 个人各下注 n_bets 次，每次投入当前资金的 f。返回终值数组（初始资金 1）。"""
    rng = np.random.default_rng(seed)
    wins = rng.random((n_people, n_bets)) < p
    step = np.where(wins, 1 + f * win, 1 - f * loss)
    if np.any(step <= 0):
        step = np.maximum(step, 0.0)
    with np.errstate(divide="ignore"):
        logw = np.log(step).sum(axis=1)
    return np.exp(np.minimum(logw, 700.0)), logw


def summarize(final, logw, n_bets):
    return {
        "mean": float(final.mean()),
        "median": float(np.median(final)),
        "frac_below_1": float((final < 1).mean()),
        "frac_below_0p1": float((final < 0.1).mean()),
        "growth_per_bet": float(np.mean(np.maximum(logw, -1e9)) / n_bets),
    }


# ---------------- 按波动调仓 ----------------

def vol_target_positions(returns, n, target_vol, rebalance_every, periods_per_year=252):
    """仓位 = min(1, 目标波动/估计波动)，不加杠杆。
    估计只用 t 之前的 n 个收益（不含当日），每 rebalance_every 期更新一次。
    前 n 期没有估计，仓位记 nan（回测时剔除）。"""
    r = np.asarray(returns, float)
    pos = np.full(len(r), np.nan)
    cur = np.nan
    for t in range(n, len(r)):
        if (t - n) % rebalance_every == 0:
            est = r[t - n:t].std(ddof=1) * np.sqrt(periods_per_year)
            cur = 1.0 if est <= 0 else min(1.0, target_vol / est)
        pos[t] = cur
    return pos


def apply_positions(returns, pos, cost_per_turnover=0.0):
    """策略日收益 = 仓位*收益 - 成本*|仓位变化|。剔除 nan 段。"""
    r = np.asarray(returns, float)
    pos = np.asarray(pos, float)
    ok = ~np.isnan(pos)
    r, pos = r[ok], pos[ok]
    turnover = np.abs(np.diff(pos, prepend=pos[0]))
    return pos * r - cost_per_turnover * turnover


def max_drawdown(strategy_returns):
    w = np.cumprod(1 + np.asarray(strategy_returns, float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], w]))[1:]
    return float((w / peak - 1).min())


def sharpe(strategy_returns, periods_per_year=252):
    s = np.asarray(strategy_returns, float)
    sd = s.std(ddof=1)
    return float(s.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def shuffle_benchmark(returns, pos, n_shuffles, seed, cost_per_turnover=0.0):
    """把仓位序列（剔除 nan 后）随机打乱再应用，返回每次的收益/波动比数组。
    保留平均仓位，去掉与当期波动的对应关系。"""
    r = np.asarray(returns, float)
    pos = np.asarray(pos, float)
    ok = ~np.isnan(pos)
    r, p = r[ok], pos[ok]
    rng = np.random.default_rng(seed)
    out = np.empty(n_shuffles)
    for i in range(n_shuffles):
        out[i] = sharpe(apply_positions(r, rng.permutation(p), cost_per_turnover))
    return out


def one_sided_p(observed, null):
    """单侧 p：null 中 >= observed 的占比（含自身，+1 平滑）。"""
    null = np.asarray(null)
    return float((1 + (null >= observed).sum()) / (1 + len(null)))
