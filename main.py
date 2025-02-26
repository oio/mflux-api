import base64
import io
import subprocess
import json
import os
import time
import random
import asyncio
from functools import lru_cache
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from PIL import Image
from mflux import Flux1, Config

USAGE_FILE = "usage.yo"
NESPRESSO_ENERGY_WH = 10.5
MACMON_PATH = "./macmon/0.5.1/bin/macmon"

# Model cache - stores loaded models for reuse
MODEL_CACHE = {}

def get_or_load_model(model_name: str, quantize: int):
    """Get a model from cache or load it if not present."""
    cache_key = f"{model_name}_{quantize}"
    if cache_key not in MODEL_CACHE:
        print(f"Loading model {model_name} with quantize={quantize}")
        MODEL_CACHE[cache_key] = Flux1.from_name(model_name, quantize)
    return MODEL_CACHE[cache_key]

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

async def save_usage_data_async(total_energy: float, image_count: int):
    """Asynchronously update usage.yo with new total_energy and image_count."""
    data = {
        "total_energy": total_energy,
        "image_count": image_count
    }
    await asyncio.to_thread(save_usage_data_sync, data)
    
def save_usage_data_sync(data: Dict[str, Any]):
    """Synchronous file writing function to be called in a separate thread."""
    with open(USAGE_FILE, "w") as f:
        json.dump(data, f)

def save_usage_data(total_energy: float, image_count: int):
    """Update usage.yo with new total_energy and image_count (synchronous fallback)."""
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

app = FastAPI(title="Mflux API", 
             description="API for generating images with mflux",
             version="0.2.0")

# Add simple queue system
request_queue = asyncio.Queue()
concurrent_requests = 0
MAX_CONCURRENT_REQUESTS = 1  # Number of concurrent requests allowed

# Start the queue processor as a background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(queue_processor())
    print("Queue processor background task started.")

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
    seed: Optional[int] = None  # Now optional, None means random seed
    prompt: str = "Luxury food photograph"
    num_inference_steps: int = 2
    height: int = 1024
    width: int = 1024
    # Keep these fields for future compatibility, but they won't be used for now
    image_prompt: Optional[str] = Field(None, description="Base64 encoded image data (not supported yet)")
    image_weight: Optional[float] = Field(0.5, description="Weight for the image prompt (not supported yet)")

async def get_macmon_metrics_async():
    """If macmon not installed, return None. Otherwise, return a dict of power usage asynchronously."""
    if not MACMON_INSTALLED:
        return None
    try:
        # Run the subprocess in a thread pool to avoid blocking the event loop
        proc = await asyncio.create_subprocess_exec(
            MACMON_PATH, "pipe", "-s", "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        data = json.loads(stdout.decode().strip())
        return {
            "cpu_power": data.get("cpu_power", 0),
            "gpu_power": data.get("gpu_power", 0),
            "ram_power": data.get("ram_power", 0),
            "all_power": data.get("all_power", 0),
        }
    except Exception as e:
        print(f"Error retrieving metrics: {e}")
        return None

def get_macmon_metrics():
    """If macmon not installed, return None. Otherwise, return a dict of power usage (synchronous fallback)."""
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

@app.get("/queue-status")
async def queue_status():
    """Return the current queue status"""
    global request_queue, concurrent_requests
    return {
        "active_requests": concurrent_requests,
        "queue_length": request_queue.qsize(),
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS
    }

# This function processes the queue continuously
async def queue_processor():
    """Background task that processes the queue"""
    global concurrent_requests
    
    while True:
        # Wait for an item from the queue
        request_tuple = await request_queue.get()
        
        # Unpack the request tuple
        req, background_tasks, response_queue = request_tuple
        
        try:
            # Update concurrent requests count
            concurrent_requests += 1
            
            # Process the request
            async for chunk in event_stream_internal(req, background_tasks):
                # Put each chunk into the response queue
                await response_queue.put(chunk)
                
        except Exception as e:
            # Handle errors
            await response_queue.put(f"data: Error during processing: {str(e)}\n\n")
            print(f"Error processing request: {str(e)}")
        finally:
            # Mark task as done
            request_queue.task_done()
            concurrent_requests -= 1
            # Signal end of response
            await response_queue.put(None)

# Internal event stream function that can be used directly or via the queue
async def event_stream_internal(req: GenerateRequest, background_tasks: BackgroundTasks = None, request_id: str = None):
    """Internal function that generates the event stream for a request"""
    global session_energy_used, total_energy_used
    global session_image_count, image_count_used
    
    try:
        yield "data: Generation started...\n\n"

        # Notify if image prompt was provided but won't be used
        if req.image_prompt:
            yield "data: Image-to-image generation is not supported in this version. Using text-only generation...\n\n"

        # (A) Load & configure MFLUX model (using cache)
        yield "data: Loading MFLUX Model...\n\n"
        flux = get_or_load_model(req.model_name, req.quantize)
        config = Config(num_inference_steps=req.num_inference_steps,
                        height=req.height, width=req.width)

        # Rest of the generation process...
        # (B) Macmon check (async)
        power_before = await get_macmon_metrics_async()
        if power_before:
            yield "data: Measuring initial power...\n\n"
        else:
            yield "data: macmon not installed, skipping power usage...\n\n"

        # (C) Time measurement
        start_time = time.time()

        # (D) Generate the image
        yield "data: Generating with text prompt...\n\n"
        
        # Use random seed if not provided
        seed = req.seed if req.seed is not None else random.randint(1, 999999)
        yield f"data: Using seed: {seed}...\n\n"
        
        generated_image = await asyncio.to_thread(
            flux.generate_image, seed, req.prompt, config=config
        )
        elapsed = time.time() - start_time
        yield f"data: Image generated ({elapsed:.2f} seconds)...\n\n"

        # (E) Post-generation power
        power_after = await get_macmon_metrics_async() if power_before else None

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
            "seed": seed,  # Include the seed that was used
        }

        # Convert image to base64
        pil_image = generated_image.image
        
        def encode_image():
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
        
        final_response["image_base64"] = await asyncio.to_thread(encode_image)

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
        if background_tasks:
            background_tasks.add_task(save_usage_data_sync, {
                "total_energy": total_energy_used, 
                "image_count": image_count_used
            })
        else:
            await save_usage_data_async(total_energy_used, image_count_used)

        yield f"data: {json.dumps(final_response)}\n\n"
    
    except Exception as e:
        yield f"data: Error during image generation: {str(e)}\n\n"

@app.post("/generate")
async def generate_image(req: GenerateRequest = Body(...), background_tasks: BackgroundTasks = None):
    """Generate an image with queue support"""
    global request_queue
    
    # Create a response queue for this specific request
    response_queue = asyncio.Queue()
    
    # Queue position - 0 is actively processing
    queue_position = request_queue.qsize()
    if queue_position > 0:
        # Not processing immediately - add queue position message
        await response_queue.put(f"data: You are position {queue_position} in the queue...\n\n")
        
        # Add message about position updates
        await response_queue.put("data: Position updates every 2 seconds...\n\n")
    
    # Add the request to the queue
    await request_queue.put((req, background_tasks, response_queue))
    print(f"Added request to queue at position {queue_position}")
    
    # Stream response from the queue
    async def stream_response():
        # Let's simplify - we'll just periodically update the queue position
        # while waiting for processing to start
        position_updates_enabled = queue_position > 0
        last_position = queue_position
        
        # Always process all chunks as they arrive
        while True:
            # Wait a bit then check position if we're still queued
            if position_updates_enabled:
                # Check for a chunk right away (non-blocking)
                try:
                    chunk = response_queue.get_nowait()
                    # If we get a chunk, we must be processing now
                    position_updates_enabled = False
                except asyncio.QueueEmpty:
                    # No chunk yet, we're still in queue
                    await asyncio.sleep(2.0)  # Wait 2 seconds
                    
                    # Check if our position has changed
                    current_position = request_queue.qsize()
                    if current_position < last_position:
                        last_position = current_position
                        if current_position > 0:
                            yield f"data: You are position {current_position} in the queue...\n\n"
                        else:
                            yield "data: Your request is being processed...\n\n"
                            position_updates_enabled = False  # Stop position updates
                    
                    # Try again
                    continue
            else:
                # Just wait for the next chunk
                chunk = await response_queue.get()
            
            # Process the chunk we received
            if chunk is None:
                # End of stream
                break
                
            # Send the chunk to the client
            yield chunk
            
            # Mark as done
            response_queue.task_done()
            
    # Return the streaming response
    return StreamingResponse(stream_response(), media_type="text/event-stream")