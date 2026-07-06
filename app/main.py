# app/main.py
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="RAG Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    """Liveness/readiness probe for load balancers and orchestrators."""
    return {"status": "ok"}


# Mounted last and at "/" so it only catches requests that didn't match
# an API route above (e.g. "/", "/index.html") — API paths like /chat
# and /upload are matched first since they were registered earlier.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
