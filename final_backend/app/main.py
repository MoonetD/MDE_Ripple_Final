from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import video

# Create FastAPI application
app = FastAPI(
    title="Video Recording API",
    description="API for starting and processing video recordings",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(video.router)

# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint that returns a welcome message.
    
    Returns:
        dict: A dictionary with a welcome message
    """
    return {
        "message": "Welcome to the Video Recording API",
        "docs": "/docs",
        "endpoints": {
            "start_record": "/video/start_record",
            "process_video": "/video/process_video"
        }
    }

# Health check endpoint
@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns:
        dict: A dictionary with the status of the API
    """
    return {"status": "healthy"}