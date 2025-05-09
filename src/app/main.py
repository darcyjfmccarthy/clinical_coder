from fastapi import FastAPI
from .routing import router
from .config import Settings

settings = Settings()         # loads .env
app = FastAPI(title="ICD10 Coder MVP")

app.include_router(router)

@app.on_event("startup")
async def startup():
    from agents.preprocessor import nlp_init
    from services.terminology import Terminology
    nlp_init()
    Terminology.connect(settings.ICD_DB_PATH)