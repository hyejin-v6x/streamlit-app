from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
from typing import List, Union, Optional, Dict, Any

from datetime import datetime
from itertools import combinations
import argparse
import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

import time
from omegaconf import OmegaConf
#from utils import seed_everything
from pytorch_lightning.utilities.seed import seed_everything
import numpy as np
from torch.utils.data import Dataset, DataLoader

# from nlp_model import Model
import streamlit as st

from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers import PNDMScheduler, EulerAncestralDiscreteScheduler
from diffusers.utils import export_to_gif
from datetime import datetime
import os

app = FastAPI()

models = {} 
model_id = "gsdf/Counterfeit-V2.5"

class ModelInput(BaseModel):
    model_id: str

class PromptInput(BaseModel):
    p_prompt: str
    n_prompt: str
    num_frames: int=16
    guidance_scale: int=13
    num_inference_steps: int=20

@app.get("/")
def hello_world():
    return {"hello": "world"}

@app.get("/model_list")
def model_list():
    return {"models": list(models.keys())}

@app.post("/load_model", description="load model")
async def load_model(data: ModelInput):
    if(data.model_id in models.keys()):
        return {"load_model": data.model_id}

    global model_id
    model_id = data.model_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    # load SD 1.5 based finetuned model
    # model_id = "gsdf/Counterfeit-V2.5"
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
    scheduler = DDIMScheduler(
        beta_schedule="scaled_linear",
        prediction_type="epsilon",    
        timestep_spacing="linspace",
        beta_start=0.00285,
        beta_end=0.012,
        steps_offset=1,
        clip_sample=False,
    )
    pipe.scheduler = scheduler

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    models[model_id] = pipe

    return {"load_model": model_id}



@app.post("/inference", description="load model")
async def inference(data: PromptInput):
    global model_id

    if(model_id not in list(models.keys())):
        return {"error": f"{model_id} is not loaded"}

    dt = datetime.now().strftime("%H-%M-%S")
    output_dir = f"/home/hyejin-voyagerx/app/output/{dt}_{model_id.split('/')[1]}_{data.p_prompt[:20]}.gif"

    print(data, model_id)
    print('-----------------')
    pipe = models[model_id]
    output = pipe(
        prompt=data.p_prompt,
        negative_prompt=data.n_prompt,
        num_frames=data.num_frames,
        guidance_scale=data.guidance_scale,
        num_inference_steps=data.num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(10788741199826055526),
    )
    frames = output.frames[0]
    export_to_gif(frames, output_dir)
    return {"output_dir": output_dir}
