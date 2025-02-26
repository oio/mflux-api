import base64
import io
import asyncio
import time
import random
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from mflux import Flux1, Config

# --- 1. Define your request schema (pydantic model for type checking) ---
class GenerateRequest(BaseModel):
    model_name: str = "schnell"         # "schnell" or "dev"
    quantize: int = 8                  # 4 or 8 (or None for full precision)
    seed: Optional[int] = None         # Now optional, None means random seed
    prompt: str = "Luxury food photograph"
    num_inference_steps: int = 2       # typical range: 2-4 for "schnell", 20-25 for "dev"
    height: int = 1024
    width: int = 1024

# --- 2. Initialize FastAPI ---
app = FastAPI(title="Simple Mflux API", 
             description="Simplified API for generating images with mflux",
             version="0.2.0")

# Add CORS middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # <-- Allows any origin
    allow_credentials=True,
    allow_methods=["*"],   # <-- Allows all methods
    allow_headers=["*"],   # <-- Allows all headers
)

# Model cache for better performance
MODEL_CACHE = {}

def get_or_load_model(model_name: str, quantize: int):
    """Get a model from cache or load it if not present"""
    cache_key = f"{model_name}_{quantize}"
    if cache_key not in MODEL_CACHE:
        print(f"Loading model {model_name} with quantize={quantize}")
        MODEL_CACHE[cache_key] = Flux1.from_name(model_name, quantize)
    return MODEL_CACHE[cache_key]

# Error handling decorator
def handle_errors(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return {"error": str(e), "success": False}
    return wrapper

# 3. Define the endpoint
@app.post("/generate")
@handle_errors
async def generate_image(req: GenerateRequest = Body(...)):
    """
    Generate an image with MFLUX and return it as base64 PNG.
    """
    # (A) Initialize the MFLUX model from cache
    flux = get_or_load_model(req.model_name, req.quantize)

    # (B) Build the config object
    config = Config(
        num_inference_steps=req.num_inference_steps,
        height=req.height,
        width=req.width,
    )

    # Record time
    start_time = time.time()

    # Use random seed if not provided
    seed = req.seed if req.seed is not None else random.randint(1, 999999)
    print(f"Using seed: {seed}")
    
    # (C) Generate the MFLUX image in a thread pool to avoid blocking
    generated_image = await asyncio.to_thread(
        flux.generate_image,
        seed=seed,
        prompt=req.prompt,
        config=config,
    )

    # Calculate elapsed time
    elapsed = time.time() - start_time

    # (D) Access the real Pillow Image inside
    pil_image = generated_image.image

    # (E) Convert it to base64-encoded PNG in a thread pool
    async def encode_image():
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    
    encoded = await asyncio.to_thread(encode_image)

    # (F) Return JSON
    return {
        "image_base64": encoded, 
        "generation_time_s": elapsed,
        "seed": seed,  # Include the seed that was used
        "success": True
    }