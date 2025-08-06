from fastapi import FastAPI
from src.routes import predict

app = FastAPI()

app.include_router(predict.router)
