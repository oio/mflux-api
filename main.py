import base64
import io
import subprocess
import json
import os
import time
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
from mflux import Flux1, Config

USAGE_FILE = "usage.yo"
NESPRESSO_ENERGY_WH = 10.5
MACMON_PATH = "./macmon/0.5.1/bin/macmon"

##########################
# Detect if macmon exists
##########################
try:
    subprocess.run([MACMON_PATH, "--version"], check=True, capture_output=True)
    MACMON_INSTALLED = True
    print("macmon detected. Power usage will be tracked.")
except Exception:
    MACMON_INSTALLED = False
    print("macmon not found. Skipping power usage tracking...")

##########################
# Load/Save usage data
##########################
def load_usage_data():
    """
    We store a JSON object in usage.yo, e.g.:
       {
         "total_energy": 12.34,
         "image_count": 5
       }
    """
    if not os.path.exists(USAGE_FILE):
        return {"total_energy": 0.0, "image_count": 0}
    try:
        with open(USAGE_FILE, "r") as f:
            data = json.load(f)
            if "total_energy" not in data:
                data["total_energy"] = 0.0
            if "image_count" not in data:
                data["image_count"] = 0
            return data
    except (ValueError, json.JSONDecodeError):
        # File is corrupted or empty, fallback
        return {"total_energy": 0.0, "image_count": 0}

def save_usage_data(total_energy: float, image_count: int):
    """Update usage.yo with new total_energy and image_count."""
    data = {
        "total_energy": total_energy,
        "image_count": image_count
    }
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)

# Read usage data at startup
usage_data = load_usage_data()
total_energy_used = usage_data["total_energy"]
image_count_used = usage_data["image_count"]

##########################
# Global session usage
##########################
session_energy_used = 0.0
session_image_count = 0  # how many images in the current session

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
    # Keep these fields for future compatibility, but they won't be used for now
    image_prompt: Optional[str] = Field(None, description="Base64 encoded image data (not supported yet)")
    image_weight: Optional[float] = Field(0.5, description="Weight for the image prompt (not supported yet)")

def get_macmon_metrics():
    """If macmon not installed, return None. Otherwise, return a dict of power usage."""
    if not MACMON_INSTALLED:
        return None
    try:
        result = subprocess.run([MACMON_PATH, "pipe", "-s", "1"], capture_output=True, text=True)
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
    global session_image_count, image_count_used

    async def event_stream():
        global session_energy_used, total_energy_used
        global session_image_count, image_count_used

        yield "data: Generation started...\n\n"

        # Notify if image prompt was provided but won't be used
        if req.image_prompt:
            yield "data: Image-to-image generation is not supported in this version. Using text-only generation...\n\n"

        # (A) Load & configure MFLUX model
        yield "data: Loading MFLUX Model...\n\n"
        flux = Flux1.from_name(req.model_name, req.quantize)
        config = Config(num_inference_steps=req.num_inference_steps,
                        height=req.height, width=req.width)

        # (B) Macmon check
        power_before = get_macmon_metrics()
        if power_before:
            yield "data: Measuring initial power...\n\n"
        else:
            yield "data: macmon not installed, skipping power usage...\n\n"

        # (C) Time measurement
        start_time = time.time()

        # (D) Generate the image
        yield "data: Generating with text prompt...\n\n"
        generated_image = flux.generate_image(req.seed, req.prompt, config=config)
        elapsed = time.time() - start_time
        yield f"data: Image generated ({elapsed:.2f} seconds)...\n\n"

        # (E) Post-generation power
        power_after = get_macmon_metrics() if power_before else None

        # (F) Prepare final response
        final_response = {
            "macmon_installed": MACMON_INSTALLED,
            "image_base64": None,
            "power_usage": None,
            "nespresso_equiv": None,
            "session_energy_used": None,
            "session_energy_nespresso": None,
            "total_energy_used": None,
            "total_energy_nespresso": None,
            "generation_time_s": elapsed,
            "session_image_count": None,
            "total_image_count": None,
        }

        # Convert image to base64
        pil_image = generated_image.image
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        final_response["image_base64"] = base64.b64encode(buf.read()).decode("utf-8")

        # (G) If macmon data
        if power_before and power_after:
            power_usage = {
                "cpu_power_used": abs(power_after["cpu_power"] - power_before["cpu_power"]),
                "gpu_power_used": abs(power_after["gpu_power"] - power_before["gpu_power"]),
                "ram_power_used": abs(power_after["ram_power"] - power_before["ram_power"]),
                "total_power_used": abs(power_after["all_power"] - power_before["all_power"]),
            }
            session_energy_used += power_usage["total_power_used"]
            total_energy_used += power_usage["total_power_used"]

            single_nespresso_equiv = round(power_usage["total_power_used"] / NESPRESSO_ENERGY_WH, 4)
            session_energy_coffees = round(session_energy_used / NESPRESSO_ENERGY_WH, 4)
            total_energy_coffees = round(total_energy_used / NESPRESSO_ENERGY_WH, 4)

            final_response["power_usage"] = power_usage
            final_response["nespresso_equiv"] = single_nespresso_equiv
            final_response["session_energy_used"] = session_energy_used
            final_response["session_energy_nespresso"] = session_energy_coffees
            final_response["total_energy_used"] = total_energy_used
            final_response["total_energy_nespresso"] = total_energy_coffees

        # (H) Increment image counts
        session_image_count += 1
        image_count_used += 1
        final_response["session_image_count"] = session_image_count
        final_response["total_image_count"] = image_count_used

        # (I) Save usage data
        # Even if macmon not installed, we still store the new image_count
        save_usage_data(total_energy_used, image_count_used)

        yield f"data: {json.dumps(final_response)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")