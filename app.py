print("APP STARTED FROM:", __file__)
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from diffusers import StableDiffusionPipeline
import torch
import io
import base64
from PIL import Image
import uuid
import os

app = FastAPI()

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("generated", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="template")

# Load model without safety filters
print("Loading realistic model...")

realistic_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

print("Loading manga model...")

manga_pipe = StableDiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

print("Loading japanese model...")

japanese_pipe = StableDiffusionPipeline.from_pretrained(
    r"/blob/main/大人の女性1.safetensors",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

print("All models loaded!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    model: str = Form("realistic"),
    steps: int = Form(20),
    guidance_scale: float = Form(7.5)
):
    if model == "manga":
        selected_pipe = manga_pipe
    elif model =="Realistic":
        selected_pipe = realistic_pipe
    else:
        selected_pipe =japanese_pipe
    
    # Generate image
    with torch.autocast("cuda"):
        image = selected_pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        ).images[0]
    
    # Save image
    image_id = str(uuid.uuid4())
    image_path = f"generated/{image_id}.png"
    image.save(image_path)
    
    # Convert to base64 for display
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return templates.TemplateResponse(
    request=request,
    name="result.html",
    context={
        "image": img_str,
        "prompt": prompt,
        "image_id": image_id
    }
)

@app.get("/download/{image_id}")
async def download(image_id: str):
    image_path = f"generated/{image_id}.png"
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png", filename=f"{image_id}.png")
    return {"error": "Image not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
