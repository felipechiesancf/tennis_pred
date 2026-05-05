import unicodedata
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import ATPPredictor


def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s)
                   if not unicodedata.combining(c))


_predictor: Optional[ATPPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    _predictor = ATPPredictor()
    yield


app = FastAPI(title='ATP Match Predictor', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class PredictRequest(BaseModel):
    p1_name: str
    p2_name: str
    surface: Literal['Hard', 'Clay', 'Grass']
    best_of_5: bool


@app.get('/health')
def health():
    if _predictor is None:
        raise HTTPException(503, 'Predictor not initialized')
    return {'status': 'ok', 'players_loaded': len(_predictor.snapshots)}


@app.get('/players')
def players():
    if _predictor is None:
        raise HTTPException(503, 'Predictor not initialized')
    # Dedup names that differ only in accents — keep the version WITH accents.
    seen = {}
    for name in _predictor.snapshots:
        norm = _strip_accents(name).lower()
        if norm not in seen:
            seen[norm] = name
        elif name != _strip_accents(name):
            seen[norm] = name
    # TODO: filter to players whose last match was in 2024+. Snapshots currently
    # have no `last_match_date` field — add it in atp_feature_engineering.ipynb
    # then filter here.
    return {'players': sorted(seen.values())}


@app.get('/player-stats/{player_name}')
def player_stats(player_name: str):
    if _predictor is None:
        raise HTTPException(503, 'Predictor not initialized')
    canonical = next(
        (k for k in _predictor.snapshots if k.lower() == player_name.lower()),
        None,
    )
    if canonical is None:
        raise HTTPException(404, f"Player '{player_name}' not found")
    s = _predictor.snapshots[canonical]
    hand_decode = {1.0: 'R', 0.0: 'L', 0.5: 'U'}
    # Snapshots store flat single-value stats, not per-surface. Frontend reuses
    # win_rate / ace_rate under whichever surface label is currently active.
    return {
        'name':         canonical,
        'age':          s.get('age'),
        'hand':         hand_decode.get(s.get('hand_enc'), 'Unknown'),
        'rank':         s.get('rank'),
        'win_rate':     s.get('win_rate_surface_12m'),
        'ace_rate':     s.get('ace_rate'),
    }


@app.post('/api/predict')
def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(503, 'Predictor not initialized')

    canonical = {}
    for label, name in (('p1', req.p1_name), ('p2', req.p2_name)):
        match, suggestions = _predictor.find_player(name)
        if match is None:
            raise HTTPException(
                404,
                f"Player {name!r} not found. Available players: {suggestions}",
            )
        canonical[label] = match

    return _predictor.predict(canonical['p1'], canonical['p2'], req.surface, req.best_of_5)
