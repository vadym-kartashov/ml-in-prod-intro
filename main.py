import joblib
from fastapi import FastAPI, HTTPException
from pydantic.v1 import BaseModel

model = joblib.load('readability-prediction-model.joblib')
vectorizer = joblib.load('readability-prediction-vectorizer.joblib')
app = FastAPI()

class PredictionRequest(BaseModel):
    excerpt: str

class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Transform the input text using the vectorizer
        transformed_text = vectorizer.transform([request.excerpt])
        # Make the prediction
        prediction = model.predict(transformed_text)[0]
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
