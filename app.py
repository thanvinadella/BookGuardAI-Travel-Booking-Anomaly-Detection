from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np, torch, pickle, json
from torch import nn

with open("preprocessing.pkl", "rb") as f:
    prep = pickle.load(f)
with open("model_config.json") as f:
    cfg = json.load(f)

FEATURES = cfg["features"]
THRESHOLD = float(cfg["threshold"])

class TravelBookingAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = TravelBookingAutoencoder(len(FEATURES))
model.load_state_dict(torch.load("bookguard_autoencoder.pt", map_location="cpu"))
model.eval()

app = FastAPI(title="BookGuard AI - Travel Booking Anomaly Detection")

class BookingInput(BaseModel):
    values: List[float]

@app.get("/")
def home():
    return {"message": "BookGuard AI Travel API running", "features": FEATURES}

@app.post("/score")
def score(payload: BookingInput):
    if len(payload.values) != len(FEATURES):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(FEATURES)} values in this order: {FEATURES}"
        )
    raw = np.asarray(payload.values, dtype=float).reshape(1, -1)
    if not np.isfinite(raw).all():
        raise HTTPException(status_code=400, detail="All feature values must be finite numbers.")
    x = prep["scaler"].transform(prep["imputer"].transform(raw))
    tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        score = torch.mean((tensor - model(tensor)) ** 2, dim=1).item()
    return {
        "anomaly_score": score,
        "threshold": THRESHOLD,
        "flag_for_review": bool(score > THRESHOLD),
        "note": "Unusual-pattern screening only. A flag is not proof of fraud."
    }
