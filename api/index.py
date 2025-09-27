import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from open_webui.main import app as main_app

# Create a new FastAPI app for Vercel
app = FastAPI(title="Float Chat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the main app
app.mount("/", main_app)

# Vercel handler
def handler(request, context):
    return app(request, context)
