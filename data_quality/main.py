from fastapi import APIRouter, Request
from loguru import logger
from data_quality.train import train
from data_quality.validate import validate
import pandas as pd
import json

router = APIRouter()


@router.post("/train/")
async def train_model(input: Request):
    input = await input.json()
    cat_cols = input["categorical_cols"]
    num_cols = input["numerical_cols"]
    # return_metrics = input["return_metrics"]
    data = pd.DataFrame(input["data"])
    model_path, metrics = train(data, cat_cols, num_cols)
    if return_metrics:
        return {"model_path": model_path, "metrics": metrics}
    return {"model_path": model_path}
