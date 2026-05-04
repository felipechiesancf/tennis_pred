"""ATP match predictor — three-model ensemble (Markov-cal + Direct).

Mirrors atp_model.ipynb's predict_match():
  serve regressors per side  -> p, q   (clipped to [0.40, 0.75])
  Markov chain over scoring  -> raw match prob
  Platt scaling              -> p_markov_cal
  direct match classifier    -> p_direct
  ensemble                   -> 0.5 * p_markov_cal + 0.5 * p_direct
"""
import json
import math
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from markov import markov_win_prob


P1_SERVE_FEATURES = [
    'p1_elo', 'p1_win_rate_last_20', 'p1_win_rate_last_5',
    'p1_win_rate_surface_12m', 'p1_serve_pct_surface',
    'p1_first_won_pct_roll', 'p1_second_won_pct_roll', 'p1_ace_rate',
    'p1_matches_prev_tourneys_7d', 'p1_sets_prev_tourneys_7d',
    'p1_days_rest', 'p1_h2h_winrate', 'p1_age', 'p1_hand_enc',
    'p2_elo', 'p2_win_rate_last_20', 'p2_win_rate_surface_12m',
    'p2_first_won_pct_roll', 'p2_second_won_pct_roll',
    'p2_ace_rate', 'p2_age', 'p2_hand_enc',
    'is_grand_slam', 'is_best_of_5', 'surface_encoded',
    'elo_diff', 'winrate_diff',
]
P2_SERVE_FEATURES = [
    'p2_elo', 'p2_win_rate_last_20', 'p2_win_rate_last_5',
    'p2_win_rate_surface_12m', 'p2_serve_pct_surface',
    'p2_first_won_pct_roll', 'p2_second_won_pct_roll', 'p2_ace_rate',
    'p2_matches_prev_tourneys_7d', 'p2_sets_prev_tourneys_7d',
    'p2_days_rest', 'p2_h2h_winrate', 'p2_age', 'p2_hand_enc',
    'p1_elo', 'p1_win_rate_last_20', 'p1_win_rate_surface_12m',
    'p1_first_won_pct_roll', 'p1_second_won_pct_roll',
    'p1_ace_rate', 'p1_age', 'p1_hand_enc',
    'is_grand_slam', 'is_best_of_5', 'surface_encoded',
    'elo_diff', 'winrate_diff',
]
DIRECT_FEATURES = [
    'elo_diff', 'rank_diff', 'age_diff', 'winrate_diff', 'serve_diff',
    'fatigue_diff', 'p1_h2h_winrate',
    'is_grand_slam', 'is_best_of_5', 'surface_encoded',
]


class ATPPredictor:
    SURFACE_TO_ENC = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    SERVE_CLIP_LO = 0.40
    SERVE_CLIP_HI = 0.75
    MODEL_VERSION = 'markov_ensemble_v1'

    def __init__(self, models_dir='./models'):
        models_dir = Path(models_dir)

        self.xgb_serve_p1 = xgb.XGBRegressor()
        self.xgb_serve_p1.load_model(str(models_dir / 'serve_model_p1.ubj'))

        self.xgb_serve_p2 = xgb.XGBRegressor()
        self.xgb_serve_p2.load_model(str(models_dir / 'serve_model_p2.ubj'))

        self.xgb_direct = xgb.XGBClassifier()
        self.xgb_direct.load_model(str(models_dir / 'direct_model.ubj'))

        with open(models_dir / 'platt_params.json') as f:
            pp = json.load(f)
        self.platt_coef = float(pp['coef'])
        self.platt_intercept = float(pp['intercept'])

        with open(models_dir / 'player_snapshots.json') as f:
            self.snapshots = json.load(f)

        self._lookup = {self._normalize(name): name for name in self.snapshots}

    @staticmethod
    def _normalize(name):
        return name.strip().lower()

    def find_player(self, name):
        """Return (canonical_name, None) on hit, (None, [closest]) on miss."""
        norm = self._normalize(name)
        canonical = self._lookup.get(norm)
        if canonical:
            return canonical, None
        close = get_close_matches(norm, self._lookup.keys(), n=10, cutoff=0.3)
        return None, [self._lookup[c] for c in close]

    def _platt(self, raw):
        raw = max(min(raw, 1 - 1e-6), 1e-6)
        logit = math.log(raw / (1 - raw))
        return 1.0 / (1.0 + math.exp(-(self.platt_coef * logit + self.platt_intercept)))

    def _build_row(self, p1_name, p2_name, surface, is_grand_slam, is_best_of_5):
        p1s = self.snapshots[p1_name]
        p2s = self.snapshots[p2_name]
        row = {
            'is_grand_slam':   int(bool(is_grand_slam)),
            'is_best_of_5':    int(bool(is_best_of_5)),
            'surface_encoded': self.SURFACE_TO_ENC.get(surface, 0),
        }
        for k, v in p1s.items():
            row[f'p1_{k}'] = float(v) if v is not None else np.nan
        for k, v in p2s.items():
            row[f'p2_{k}'] = float(v) if v is not None else np.nan
        row['elo_diff']     = row['p1_elo']               - row['p2_elo']
        row['rank_diff']    = row['p1_rank']              - row['p2_rank']
        row['age_diff']     = row['p1_age']               - row['p2_age']
        row['winrate_diff'] = row['p1_win_rate_last_20']  - row['p2_win_rate_last_20']
        row['serve_diff']   = row['p1_serve_pct_surface'] - row['p2_serve_pct_surface']
        row['fatigue_diff'] = (row['p1_matches_prev_tourneys_7d']
                               - row['p2_matches_prev_tourneys_7d'])
        return row

    def predict(self, p1_name, p2_name, surface, best_of_5):
        # Men's BO5 only happens at slams in modern tour play.
        is_grand_slam = bool(best_of_5)
        row = self._build_row(p1_name, p2_name, surface, is_grand_slam, best_of_5)
        one = pd.DataFrame([row])
        X1 = one[P1_SERVE_FEATURES].astype(float).values
        X2 = one[P2_SERVE_FEATURES].astype(float).values
        Xdir = one[DIRECT_FEATURES].astype(float).values

        p_serve_p1 = float(np.clip(self.xgb_serve_p1.predict(X1)[0],
                                   self.SERVE_CLIP_LO, self.SERVE_CLIP_HI))
        p_serve_p2 = float(np.clip(self.xgb_serve_p2.predict(X2)[0],
                                   self.SERVE_CLIP_LO, self.SERVE_CLIP_HI))

        raw = markov_win_prob(p_serve_p1, p_serve_p2, best_of_5=bool(best_of_5))
        prob = self._platt(raw)
        direct_prob = float(self.xgb_direct.predict_proba(Xdir)[0, 1])
        final_prob = 0.5 * prob + 0.5 * direct_prob

        return {
            'p1_win_prob': final_prob,
            'p2_win_prob': 1.0 - final_prob,
            'predicted_winner': p1_name if final_prob > 0.5 else p2_name,
            'model': self.MODEL_VERSION,
            'surface': surface,
        }
