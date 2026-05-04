"""Markov chain over the tennis scoring tree.

p = server win prob on p1's serve, q = server win prob on p2's serve.
Closed-form at game level, memoized DP for set/match.
"""
from functools import lru_cache


def _game_prob(s):
    if s <= 0:
        return 0.0
    if s >= 1:
        return 1.0
    sq = 1.0 - s
    denom = s * s + sq * sq
    deuce = (s * s) / denom if denom > 0 else 0.5
    return s ** 4 + 4 * s ** 4 * sq + 10 * s ** 4 * sq ** 2 + 20 * s ** 3 * sq ** 3 * deuce


def _tiebreak_prob(p, q, max_points=50):
    cache = {}
    for total in range(max_points, -1, -1):
        for s1 in range(total + 1):
            s2 = total - s1
            if s1 >= 7 and s1 - s2 >= 2:
                cache[(s1, s2)] = 1.0
                continue
            if s2 >= 7 and s2 - s1 >= 2:
                cache[(s1, s2)] = 0.0
                continue
            if total == max_points:
                cache[(s1, s2)] = 0.5
                continue
            idx = total
            if idx == 0:
                win_p1_pt = p
            else:
                pair = (idx - 1) // 2
                win_p1_pt = (1.0 - q) if pair % 2 == 0 else p
            cache[(s1, s2)] = (win_p1_pt * cache[(s1 + 1, s2)]
                               + (1.0 - win_p1_pt) * cache[(s1, s2 + 1)])
    return cache[(0, 0)]


def markov_win_prob(p, q, best_of_5=False):
    p = max(min(float(p), 0.999), 0.001)
    q = max(min(float(q), 0.999), 0.001)
    g_p = _game_prob(p)
    g_q_loss = 1.0 - _game_prob(q)
    tb_prob = _tiebreak_prob(p, q)

    @lru_cache(maxsize=None)
    def set_dp(g1, g2, p1_serves):
        if g1 == 6 and g2 <= 4:
            return 1.0
        if g2 == 6 and g1 <= 4:
            return 0.0
        if g1 == 7 and g2 in (5, 6):
            return 1.0
        if g2 == 7 and g1 in (5, 6):
            return 0.0
        if g1 == 6 and g2 == 6:
            return tb_prob
        win_game = g_p if p1_serves else g_q_loss
        return (win_game * set_dp(g1 + 1, g2, not p1_serves)
                + (1.0 - win_game) * set_dp(g1, g2 + 1, not p1_serves))

    set_p = set_dp(0, 0, True)
    target = 3 if best_of_5 else 2

    @lru_cache(maxsize=None)
    def match_dp(s1, s2):
        if s1 == target:
            return 1.0
        if s2 == target:
            return 0.0
        return set_p * match_dp(s1 + 1, s2) + (1.0 - set_p) * match_dp(s1, s2 + 1)

    return match_dp(0, 0)
