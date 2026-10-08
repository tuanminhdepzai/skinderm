import modal
import os
import sys

# Create the Modal App
app = modal.App("skinderm-backend")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install system dependencies for OpenCV and other packages
    .apt_install("libgl1", "libglib2.0-0")
    # Set working directory to /root where the code resides
    .workdir("/root")
    # Install Python packages
    .pip_install(
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-multipart==0.0.6",
        "pydantic>=2.7.0",
        "torch",
        "torchvision",
        "timm",
        "segmentation-models-pytorch",
        "albumentations",
        "opencv-python-headless",
        "matplotlib",
        "firebase-admin",
        "supabase",
        "python-dotenv",
        "openai",
        "apscheduler",
        "langgraph",
        "langchain",
        "requests",
        "pillow",
        "imagekitio"
    )
    # Add local directory contents to /root at container startup (copy=False ensures latest code is mounted)
    .add_local_dir(
        ".",
        remote_path="/root",
        copy=False,
        ignore=["**/__pycache__", "**/.env", "**/.git", "**/skinderm.db", "**/skinderm_storage"]
    )
)

# Define a persistent volume for the database and storage files
# We mount this at /data.
volume = modal.Volume.from_name("skinderm-storage", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_dotenv()],
    timeout=600,
    cpu=2.0,
    memory=8192,
)
@modal.asgi_app()
def fastapi_app():
    # Set STORAGE_ROOT to point to the persistent volume mount path
    os.environ["STORAGE_ROOT"] = "/data"
    
    # Ensure directory containing main.py is in sys.path and is current working directory
    if os.path.exists("/root/backend/main.py"):
        os.chdir("/root/backend")
        if "/root/backend" not in sys.path:
            sys.path.insert(0, "/root/backend")
    elif os.path.exists("/root/main.py"):
        os.chdir("/root")
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
    
    # Import the FastAPI app inside the function context
    from main import app as web_app
    return web_app
