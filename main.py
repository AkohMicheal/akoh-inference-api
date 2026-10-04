from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as predict_router

app = FastAPI(
    title="Akoh Inference API",
    description="Production-ready FastAPI ML inference microservice serving clinical diagnosis and industrial predictive maintenance models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for client/dashboard consumption
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Akoh Inference API",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": [
            {"path": "/predict/heart-disease", "method": "POST", "target": "UCI Heart Disease"},
            {"path": "/predict/maintenance", "method": "POST", "target": "AI4I Predictive Maintenance"},
            {"path": "/health", "method": "GET", "target": "System Health"}
        ]
    }


@app.get("/health", tags=["General"])
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "heart_disease": "ready",
            "predictive_maintenance": "ready"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
