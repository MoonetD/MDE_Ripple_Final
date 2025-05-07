import cv2
import time
import mss
import numpy as np
import win32gui
import pathlib
import uvicorn  # Import uvicorn
import threading # For the stop signal event

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel

# --- Configuration ---
TITLE_SUBSTR = "O-KAM"          # <--- Part of the window title to capture (CASE-INSENSITIVE)
FPS          = 20               # Frame rate (adjust as needed)
WEBCAM_ID    = 1                # Default webcam ID

# --- Global State (for managing the single recording session) ---
# Using a dictionary to hold state is slightly cleaner than loose globals
recording_state = {
    "is_recording": False,
    "stop_event": threading.Event(), # Event to signal stopping
    "screen_writer": None,
    "webcam_writer": None,
    "webcam_capture": None,
    "output_path": None,
    "user_id": None,
    "screen_file": None,
    "webcam_file": None,
}

# --- Helper Functions ---
def find_window_rect(title_substr: str):
    """Return (left, top, right, bottom) of the first visible window whose
    title contains title_substr (case-insensitive)."""
    matches = []
    def cb(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if title_substr.lower() in window_title.lower():
                matches.append(win32gui.GetWindowRect(hwnd))
    try:
        win32gui.EnumWindows(cb, None)
    except Exception:
        # Handle potential exceptions during window enumeration if needed
        pass
    return matches[0] if matches else None

def recording_task(window_rect, screen_file, webcam_file, fps, webcam_id):
    """The actual recording loop that runs in the background."""
    l, t, r, b = window_rect
    w, h = r - l, b - t

    try:
        # Initialize screen recording writer
        screen_vw = cv2.VideoWriter(str(screen_file),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps, (w, h))
        recording_state["screen_writer"] = screen_vw # Store globally

        # Initialize webcam
        webcam = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW) # Use CAP_DSHOW for better compatibility on Windows
        if not webcam.isOpened():
            print(f"❌ ERROR: Could not open webcam ID {webcam_id}")
            # Signal that recording setup failed (though the endpoint already returned success)
            # More robust error handling might involve IPC or a status check endpoint
            recording_state["is_recording"] = False # Reset flag
            screen_vw.release() # Release screen writer
            return # Stop the task

        recording_state["webcam_capture"] = webcam # Store globally
        webcam_w = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        webcam_h = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize webcam recording writer
        webcam_vw = cv2.VideoWriter(str(webcam_file),
                                    cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps, (webcam_w, webcam_h))
        recording_state["webcam_writer"] = webcam_vw # Store globally

        print(f"▶️  Recording task started: Screen ({w}x{h}) and Webcam ({webcam_w}x{webcam_h})")
        print(f"   Screen File: {screen_file}")
        print(f"   Webcam File: {webcam_file}")

        interval = 1.0 / fps
        last_frame_time = time.perf_counter()

        with mss.mss() as sct:
            monitor = {"left": l, "top": t, "width": w, "height": h}
            while not recording_state["stop_event"].is_set():
                current_time = time.perf_counter()
                if current_time - last_frame_time < interval:
                    time.sleep(max(0, interval - (current_time - last_frame_time) - 0.001)) # More accurate sleep
                    continue # Ensure we don't exceed FPS significantly

                last_frame_time = time.perf_counter() # Update time after potential sleep

                # Capture screen
                screen_frame_bgra = sct.grab(monitor)
                screen_frame_bgr = cv2.cvtColor(np.array(screen_frame_bgra), cv2.COLOR_BGRA2BGR)
                screen_vw.write(screen_frame_bgr)

                # Capture webcam
                ret, webcam_frame = webcam.read()
                if ret:
                    webcam_vw.write(webcam_frame)
                #else:
                #    print("Warning: Failed to capture webcam frame") # Optional warning

        print("ℹ️  Stop signal received, finishing recording...")

    except Exception as e:
        print(f"❌ ERROR during recording loop: {e}")
        # Attempt to clean up even if an error occurs mid-loop
    finally:
        print("⏹️  Releasing resources...")
        if recording_state["screen_writer"]:
            recording_state["screen_writer"].release()
            print("   Screen writer released.")
        if recording_state["webcam_writer"]:
            recording_state["webcam_writer"].release()
            print("   Webcam writer released.")
        if recording_state["webcam_capture"]:
            recording_state["webcam_capture"].release()
            print("   Webcam capture released.")

        # Reset global state
        recording_state["is_recording"] = False
        recording_state["stop_event"].clear() # Clear the event for next time
        recording_state["screen_writer"] = None
        recording_state["webcam_writer"] = None
        recording_state["webcam_capture"] = None
        print(f"✅ Recording stopped. Files saved:")
        print(f"   Screen: {recording_state.get('screen_file', 'N/A')}")
        print(f"   Webcam: {recording_state.get('webcam_file', 'N/A')}")
        recording_state["output_path"] = None
        recording_state["user_id"] = None
        recording_state["screen_file"] = None
        recording_state["webcam_file"] = None


# --- FastAPI Setup ---
app = FastAPI(
    title="Window and Webcam Recorder API",
    description="API to start and stop recording a specific window and the webcam.",
    version="1.0.0"
)

class StartRecordingRequest(BaseModel):
    output_path: str
    user_id: str

@app.post("/start_recording", status_code=202) # 202 Accepted: request accepted, processing started
async def start_recording(request: StartRecordingRequest, background_tasks: BackgroundTasks):
    """
    Starts recording the specified window and the default webcam.
    Saves files to {output_path}/{user_id}_screen_record.mp4 and
    {output_path}/{user_id}_webcam.mp4.
    """
    if recording_state["is_recording"]:
        raise HTTPException(status_code=409, detail="Recording is already in progress.")

    # Find the window
    rect = find_window_rect(TITLE_SUBSTR)
    if not rect:
        raise HTTPException(status_code=404, detail=f"Window containing '{TITLE_SUBSTR}' not found or not visible.")

    # Prepare output directory and filenames
    out_dir = pathlib.Path(request.output_path)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create output directory: {e}")

    screen_file = out_dir / f"{request.user_id}_screen_record.mp4"
    webcam_file = out_dir / f"{request.user_id}_webcam.mp4"

    # Update global state BEFORE starting background task
    recording_state["is_recording"] = True
    recording_state["stop_event"].clear() # Ensure stop event is clear
    recording_state["output_path"] = request.output_path
    recording_state["user_id"] = request.user_id
    recording_state["screen_file"] = screen_file
    recording_state["webcam_file"] = webcam_file

    # Add the recording function to background tasks
    background_tasks.add_task(
        recording_task,
        window_rect=rect,
        screen_file=screen_file,
        webcam_file=webcam_file,
        fps=FPS,
        webcam_id=WEBCAM_ID
    )

    return {"message": f"Recording started for user '{request.user_id}'. Target window: '{TITLE_SUBSTR}'. Use /stop_recording to end."}

@app.post("/stop_recording")
async def stop_recording():
    """
    Stops the currently active recording session.
    """
    if not recording_state["is_recording"]:
        raise HTTPException(status_code=400, detail="No recording is currently in progress.")

    if recording_state["stop_event"].is_set():
         raise HTTPException(status_code=400, detail="Stop signal already sent.")

    print("ℹ️  Received stop request. Signaling recording task to stop...")
    recording_state["stop_event"].set() # Signal the background task to stop

    # Note: We don't wait here. The background task will clean up.
    # You could add a short sleep/wait loop if immediate confirmation is needed,
    # but it complicates the API response.

    return {"message": "Stop signal sent to recording task. Files will be saved shortly."}

@app.get("/status")
async def get_status():
    """
    Returns the current recording status.
    """
    if recording_state["is_recording"]:
        # Check if the stop event has been set but the task might still be cleaning up
        status = "Stopping" if recording_state["stop_event"].is_set() else "Recording"
        return {
            "status": status,
            "user_id": recording_state["user_id"],
            "output_path": recording_state["output_path"],
            "screen_file": str(recording_state["screen_file"]),
            "webcam_file": str(recording_state["webcam_file"]),
        }
    else:
        return {"status": "Idle"}


# --- Run the server ---
if __name__ == "__main__":
    print("--- Starting FastAPI Recorder ---")
    print(f"Target Window Substring: '{TITLE_SUBSTR}'")
    print(f"Webcam ID: {WEBCAM_ID}")
    print(f"FPS: {FPS}")
    print(f"Run `uvicorn main:app --host 0.0.0.0 --port 9005` or start this script.")
    # Note: Binding to 0.0.0.0 makes it accessible on your network
    # Use "127.0.0.1" for localhost only access.
    uvicorn.run(app, host="0.0.0.0", port=8005)