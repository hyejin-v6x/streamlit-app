# import io
import os

# from pathlib import Path

import requests

# from PIL import Image
# from pydantic import BaseModel

import streamlit as st
import json
import shutil

# import re
# from datetime import datetime

# path = "http://localhost:8001"
path = "https://13ee-211-109-189-50.ngrok-free.app"

if st.button("hello"):
    response = requests.get(f"{path}/")
    st.write(response.json())

if st.button("model list"):
    response = requests.get(f"{path}/model_list")
    st.write(response.json())

model_id = st.text_input("model_id", value="gsdf/Counterfeit-V2.5", disabled=True)
if st.button("load model"):
    with st.spinner(text="In progress..."):
        response = requests.post(
            f"{path}/load_model", data=json.dumps({"model_id": model_id})
        )
    st.write(response.json())


p_prompt = st.text_input(
    "p_prompt",
    value="((masterpiece,best quality)),1girl, bangs, blue eyes, blurry background, branch, brown hair, dappled sunlight, flower, from side, hair flower, hair ornament, japanese clothes, kimono, leaf, (maple leaf:1.9), obi, outdoors, sash, solo, sunlight, upper body",
    # max_chars=77,
)
n_prompt = st.text_input(
    "n_prompt",
    value="EasyNegative, letterboxed,  extra fingers,fewer fingers, bad quality, worse quality",
    # max_chars=77,
)
# num_frames = st.text_input("num_frames")
guidance_scale = st.text_input("guidance_scale", value="13")
num_inference_steps = st.text_input("num_inference_steps", value="10")


if st.button("inference!"):
    data = {
        "p_prompt": p_prompt,
        "n_prompt": n_prompt,
        "num_frames": 16,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
    }
    st.write(data)
    with st.spinner(text="In progress..."):
        response = requests.post(
            f"{path}/inference", data=json.dumps(data), stream=True
        )
    # res = response.json()
    st.write("response: ", response.status_code)
    if response.status_code == 200:
        # p_prompt = re.sub(r"[\W_]+", "", p_prompt)
        # dt = datetime.now().strftime("%H-%M-%S")
        # save_path = f"./output/{dt}_{model_id.split('/')[1]}_{p_prompt[:20]}_gs{guidance_scale}_i{num_inference_steps}.gif"
        save_path = "./output/output.gif"
        with open(save_path, "wb") as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
            st.write("done")
            if os.path.isfile(save_path):
                st.image(save_path)
    # save_dir = list(res.values())[0]
    # if os.path.isfile(save_dir):
    #     st.image(save_dir)
