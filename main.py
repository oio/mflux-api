import base64
import io
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from mflux import Flux1, Config

# --- 1. Define your request schema (pydantic model for type checking) ---
class GenerateRequest(BaseModel):
    model_name: str = "schnell"         # "schnell" or "dev"
    quantize: int = 8                  # 4 or 8 (or None for full precision)
    seed: int = 2
    prompt: str = "Luxury food photograph"
    num_inference_steps: int = 2       # typical range: 2-4 for "schnell", 20-25 for "dev"
    height: int = 1024
    width: int = 1024

# --- 2. Initialize FastAPI ---
app = FastAPI()

# Add CORS middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # <-- Allows any origin
    allow_credentials=True,
    allow_methods=["*"],   # <-- Allows all methods
    allow_headers=["*"],   # <-- Allows all headers
)

# Optional: you can load models into memory once at startup to avoid repeated loading
# However, for large models, it can be memory-intensive to keep multiple models around.
# If you only need one model, load it globally here:
# 
#   flux_global = Flux1.from_name(
#       model_name="schnell",
#       quantize=8,
#   )
# 
# Then inside the endpoint, you reuse flux_global.

# 3. Define the endpoint
@app.post("/generate")
def generate_image(req: GenerateRequest = Body(...)):
    """
    Generate an image with MFLUX and return it as base64 PNG.
    """

    # (A) Initialize the MFLUX model
    flux = Flux1.from_name(
        model_name=req.model_name,
        quantize=req.quantize,   # 4, 8, or None
    )

    # (B) Build the config object
    config = Config(
        num_inference_steps=req.num_inference_steps,
        height=req.height,
        width=req.width,
    )

    # (C) Generate the MFLUX image (a "GeneratedImage" object, not a Pillow Image)
    generated_image = flux.generate_image(
        seed=req.seed,
        prompt=req.prompt,
        config=config,
    )

    # (D) Access the real Pillow Image inside
    pil_image = generated_image.image  # <--- This is the PIL.Image.Image

    # (E) Convert it to base64-encoded PNG
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")   # Now this works, because it's a genuine Pillow Image
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("utf-8")

    # (F) Return JSON
    return {"image_base64": encoded}