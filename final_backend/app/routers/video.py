from fastapi import APIRouter, HTTPException
import time

router = APIRouter(
    prefix="/video",
    tags=["video"],
    responses={404: {"description": "Not found"}},
)

# Variable to track recording state (for demonstration purposes only)
recording = False

@router.post("/start_record")
async def start_record():
    """
    Start recording video.
    
    This endpoint simulates starting a video recording process.
    In a real implementation, this would initiate camera recording.
    
    Returns:
        dict: A dictionary with a success boolean
    """
    global recording
    recording = True
    return {"success": True}

@router.post("/process_video")
async def process_video():
    """
    Process the recorded video.
    
    This endpoint simulates stopping the recording and processing the video.
    It includes a 2-second pause to simulate processing time.
    
    Returns:
        dict: A dictionary with a success boolean
    """
    global recording
    
    # Check if recording was started (for demonstration purposes)
    if not recording:
        # In a real implementation, you might want to return an error
        # but for simplicity, we'll just set recording to False and continue
        pass
    
    # Simulate video processing with a 2-second pause
    time.sleep(2)
    
    # Reset recording state
    recording = False
    
    return {"success": True}