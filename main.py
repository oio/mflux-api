import base64
import io
import subprocess
import json
import os
import asyncio
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
from mflux import Flux1, Config

# --- Constants ---
NESPRESSO_ENERGY_WH = 10.5
USAGE_FILE = "usage.yo"

# --- Load total energy consumption ---
def load_total_energy():
    if os.path.exists(USAGE_FILE):
        with open(USAGE_FILE, "r") as f:
            try:
                return float(f.read().strip())
            except ValueError:
                return 0.0
    return 0.0

def save_total_energy(energy):
    with open(USAGE_FILE, "w") as f:
        f.write(f"{energy}")

# Global session tracking
session_energy_used = 0.0
total_energy_used = load_total_energy()

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Model ---
class GenerateRequest(BaseModel):
    model_name: str = "schnell"
    quantize: int = 8
    seed: int = 2
    prompt: str = "Luxury food photograph"
    num_inference_steps: int = 2
    height: int = 1024
    width: int = 1024

# --- Function to Get Power Metrics ---
def get_macmon_metrics():
    try:
        result = subprocess.run(["macmon", "pipe", "-s", "1"], capture_output=True, text=True)
        data = json.loads(result.stdout.strip())
        return {
            "cpu_power": data.get("cpu_power", 0),
            "gpu_power": data.get("gpu_power", 0),
            "ram_power": data.get("ram_power", 0),
            "all_power": data.get("all_power", 0),
        }
    except Exception as e:
        print(f"Error retrieving metrics: {e}")
        return None

# --- Streaming Image Generation ---
@app.post("/generate")
async def generate_image(req: GenerateRequest = Body(...)):
    global session_energy_used, total_energy_used  # <-- Declare globals here

    async def event_stream():
        global session_energy_used, total_energy_used  # <-- Declare inside async function too!

        yield "data: Generation started...\n\n"

        # (A) Get initial power metrics
        power_before = get_macmon_metrics()
        yield "data: Measuring initial power...\n\n"

        # (B) Load MFLUX Model
        flux = Flux1.from_name(req.model_name, req.quantize)
        yield "data: Model loaded...\n\n"

        # (C) Configure Model
        config = Config(num_inference_steps=req.num_inference_steps, height=req.height, width=req.width)

        # (D) Generate Image
        generated_image = flux.generate_image(req.seed, req.prompt, config=config)
        yield "data: Image generated...\n\n"

        # (E) Get final power metrics
        power_after = get_macmon_metrics()

        # (F) Compute Power Usage
        power_usage = {
            "cpu_power_used": abs(power_after["cpu_power"] - power_before["cpu_power"]),
            "gpu_power_used": abs(power_after["gpu_power"] - power_before["gpu_power"]),
            "ram_power_used": abs(power_after["ram_power"] - power_before["ram_power"]),
            "total_power_used": abs(power_after["all_power"] - power_before["all_power"]),
        }

        # (G) Update total energy usage
        session_energy_used += power_usage["total_power_used"]  # 🔥 FIX: No more UnboundLocalError
        total_energy_used += power_usage["total_power_used"]
        save_total_energy(total_energy_used)

        # (H) Compute Nespresso Equivalents
        nespresso_equiv = round(power_usage["total_power_used"] / NESPRESSO_ENERGY_WH, 4)

        # (I) Convert Image to Base64
        pil_image = generated_image.image
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        encoded_image = base64.b64encode(buf.read()).decode("utf-8")

        # (J) Send Final Response
        final_response = {
            "image_base64": encoded_image,
            "power_usage": power_usage,
            "nespresso_equiv": nespresso_equiv,
            "session_energy_used": session_energy_used,
            "total_energy_used": total_energy_used,
        }
        yield f"data: {json.dumps(final_response)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
