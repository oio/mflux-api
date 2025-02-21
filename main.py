import base64
import io
import subprocess
import json
import os
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image
from mflux import Flux1, Config

NESPRESSO_ENERGY_WH = 10.5
USAGE_FILE = "usage.yo"

##########################
# Detect if macmon exists
##########################
try:
    subprocess.run(["macmon", "--version"], check=True, capture_output=True)
    MACMON_INSTALLED = True
    print("macmon detected. Power usage will be tracked.")
except Exception:
    MACMON_INSTALLED = False
    print("macmon not found. Skipping power usage tracking...")

##########################
# Load total energy usage
##########################
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

# Global session usage
session_energy_used = 0.0
total_energy_used = load_total_energy()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    model_name: str = "schnell"
    quantize: int = 8
    seed: int = 2
    prompt: str = "Luxury food photograph"
    num_inference_steps: int = 2
    height: int = 1024
    width: int = 1024

def get_macmon_metrics():
    """
    If macmon is not installed, return None.
    Otherwise, return a dict of power usage.
    """
    if not MACMON_INSTALLED:
        return None
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

@app.post("/generate")
async def generate_image(req: GenerateRequest = Body(...)):
    global session_energy_used, total_energy_used

    async def event_stream():
        global session_energy_used, total_energy_used

        yield "data: Generation started...\n\n"

        # (A) Load & configure the MFLUX model
        yield "data: Loading MFLUX Model...\n\n"
        flux = Flux1.from_name(req.model_name, req.quantize)
        config = Config(num_inference_steps=req.num_inference_steps, 
                        height=req.height, width=req.width)

        # (B) If macmon is installed, measure power BEFORE generation
        power_before = get_macmon_metrics()
        if power_before:
            yield "data: Measuring initial power...\n\n"
        else:
            yield "data: macmon not installed, skipping power usage...\n\n"

        # (C) Generate the image
        generated_image = flux.generate_image(req.seed, req.prompt, config=config)
        yield "data: Image generated...\n\n"

        # (D) If macmon is installed, measure power AFTER generation
        power_after = get_macmon_metrics() if power_before else None

        # Prepare final response
        final_response = {
            "macmon_installed": MACMON_INSTALLED,
            "image_base64": None,         # We'll fill in the base64
            "power_usage": None,         # We’ll fill if macmon installed
            "nespresso_equiv": None,     # We’ll fill if macmon installed
            "session_energy_used": None, # We’ll fill if macmon installed
            "total_energy_used": None,   # We’ll fill if macmon installed
        }

        # (E) Convert the image to base64
        pil_image = generated_image.image
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        final_response["image_base64"] = base64.b64encode(buf.read()).decode("utf-8")

        # (F) If we have power metrics
        if power_before and power_after:
            power_usage = {
                "cpu_power_used": abs(power_after["cpu_power"] - power_before["cpu_power"]),
                "gpu_power_used": abs(power_after["gpu_power"] - power_before["gpu_power"]),
                "ram_power_used": abs(power_after["ram_power"] - power_before["ram_power"]),
                "total_power_used": abs(power_after["all_power"] - power_before["all_power"]),
            }
            session_energy_used += power_usage["total_power_used"]
            total_energy_used += power_usage["total_power_used"]
            save_total_energy(total_energy_used)

            nespresso_equiv = round(power_usage["total_power_used"] / NESPRESSO_ENERGY_WH, 4)

            final_response["power_usage"] = power_usage
            final_response["nespresso_equiv"] = nespresso_equiv
            final_response["session_energy_used"] = session_energy_used
            final_response["total_energy_used"] = total_energy_used

        # (G) Yield final JSON
        yield f"data: {json.dumps(final_response)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
