import os
import pickle
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(
    title="Accident Claim Insurance API",
    description="FastAPI service for predicting claim outcomes with visualization dashboard",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

MODEL_PATH = "models/insurance_claim_model.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as file:
        saved_data = pickle.load(file)
        model = saved_data.get('model')
        feature_names = saved_data.get('feature_names')
else:
    model = None
    feature_names = []

@app.get("/", response_class=HTMLResponse)
def read_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_claim(claims: List[Dict[str, Any]]):
    if not model:
        raise HTTPException(status_code=500, detail="Model file not loaded. Please train model first.")
    
    try:
        df = pd.DataFrame(claims)
        
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_names]
        
        predictions = model.predict(df)
        probabilities = model.predict_proba(df).tolist()
        
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
