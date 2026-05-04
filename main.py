from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import ATPPredictor


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
