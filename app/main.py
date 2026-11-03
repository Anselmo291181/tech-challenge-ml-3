import time
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from app.model_loader import InferenceEngine

app = FastAPI(
    title="API de Triagem Médica Automatizada",
    description="Classificação NLP de laudos médicos para urgência (normal, atenção, urgente).",
    version="1.0.0"
)

REQUEST_COUNT = Counter("http_requests_total", "Total de requisições HTTP", ["method", "endpoint", "status_code"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "Tempo de resposta em segundos", ["endpoint"])
PREDICTION_COUNT = Counter("model_predictions_total", "Total de predições", ["urgencia", "engine"])

engine = InferenceEngine()

class LaudoRequest(BaseModel):
    laudo_medico: str = Field(..., example="Paciente com dor torácica severa e dispneia.")

class LaudoResponse(BaseModel):
    urgencia: str
    confianca: float
    engine_utilizado: str
    tempo_execucao_ms: float

@app.get("/health")
def health_check():
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code="200").inc()
    return {"status": "ok", "onnx_active": engine.use_onnx}

@app.post("/predict", response_model=LaudoResponse)
def predict_urgency(payload: LaudoRequest):
    start_time = time.perf_counter()
    endpoint = "/predict"
    
    if not payload.laudo_medico.strip():
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status_code="400").inc()
        raise HTTPException(status_code=400, detail="O laudo médico não pode estar vazio.")
        
    try:
        urgencia, confianca, engine_type = engine.predict(payload.laudo_medico)
        latency = time.perf_counter() - start_time
        
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status_code="200").inc()
        PREDICTION_COUNT.labels(urgencia=urgencia, engine=engine_type).inc()
        
        return LaudoResponse(
            urgencia=urgencia,
            confianca=round(confianca, 4),
            engine_utilizado=engine_type,
            tempo_execucao_ms=round(latency * 1000, 2)
        )
    except Exception as e:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status_code="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)