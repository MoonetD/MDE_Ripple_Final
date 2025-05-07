import cv2
import os
import json
import shutil
from datetime import datetime
from openai import OpenAI
import base64
import logging
import json
import numpy as np
from PIL import Image
import sys

def analyze_and_save_frame(frame, frame_path, client, analysis_data):
    """Analyze a frame and save the annotated version and analysis data"""
    # Save original frame
    cv2.imwrite(frame_path, frame)
    
    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    # Get frame timestamp from filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    # Save previous frame's glass info
    prev_glass_info = None
    if analysis_data['frames']:
        prev_frames = sorted(analysis_data['frames'].items(), key=lambda x: x[0])
        if prev_frames:
            prev_glass_info = prev_frames[-1][1]
    
    # Analyze using OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are analysing a *video* frame-by-frame in chronological order.  \nMaintain an internal \"current glass ID\" that carries over from the previous frame unless you deliberately switch it (see rules 1-3).\n\n──────────────────────\n📌 1.  WHICH GLASS TO TRACK\n──────────────────────\na.  If one or more glasses are visible, pick **one**:\n    • the glass whose bounding box centre is closest to any visible hand, **or**  \n    • if multiple tie, the glass that was already being tracked in the previous frame.\nb.  If *no* glass is visible, keep the last tracked glass in memory but mark `glass_visible=false` for this frame.\n\n──────────────────────\n📌 2.  APPEARS_DIFFERENT?\n──────────────────────\nOnly set `appears_different=true` when the newly chosen glass differs *substantially* from the remembered one in material, shape, or size.  \nMinor changes caused by perspective or lighting ⇒ `false`.\n\n──────────────────────\n📌 3.  DRINKING_EVENT LOGIC\n──────────────────────\nMark `drinking_event=true` in *any* of the following situations **while a hand is gripping or clearly guiding the glass**:\n    i.   The glass is tilted > 20 ° from vertical.  \n    ii.  The water level is visibly disturbed / mid-motion.  \n    iii. The glass was visible and tracked in the *previous* frame, is **not** visible in the *current* frame (`glass_visible=false`), and appears again within the next 2 seconds **with roughly the same fill level** (±5 %) — treat the moment it left the frame as the drinking event.\n\nDo **not** mark a drinking event for horizontal hand movements, light touches, or when the glass is put down on a table without tilt or water disturbance.\n\n──────────────────────\n📌 4.  OUTPUT FORMAT  (exact JSON)\n──────────────────────\nIf **no** glass is visible *and* no previous glass is being tracked, output:\n{\"no_glass\": true}\n\nOtherwise output:\n{\n  \"no_glass\": false,\n  \"glass_visible\": boolean,        // true if glass in this frame, false if it has momentarily left the frame\n  \"left\": [x1, y1],               // pixel coords of left edge of water line; null if glass_visible=false\n  \"right\": [x2, y2],               // pixel coords of right edge of water line; null if glass_visible=false\n  \"percentage\": number,            // 0-100; -1 if glass_visible=false\n  \"glass_characteristics\": string, // brief free-text\n  \"appears_different\": boolean,\n  \"drinking_event\": boolean\n}\n\nUse null or -1 only when that field cannot be measured because the glass is out of frame in this specific image.\n\n────────────────────── 📌 5. IMPORTANT DETAILS ──────────────────────\n\nAll coordinates are in the current image's pixel space.\n\nTreat the \"next 2 seconds\" window in rule 3 iii as ≈ 50 frames at 25 fps (adjust if actual fps known).\n\nBe conservative—if uncertain, return drinking_event=false."                        
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )
        # Clean up the response and parse JSON
        response_text = response.choices[0].message.content.strip()
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            return None
    except Exception as e:
        print(f"Error making API call: {e}")
        return None
    
    # Create annotated version of the frame
    annotated_frame = frame.copy()
    
    # Save frame data
    frame_data = {
        'timestamp': timestamp,
        'frame_path': frame_path,
        'result': result,
        'has_glass': not result.get('no_glass', True),
        'glass_visible': result.get('glass_visible', False) if not result.get('no_glass', True) else False
    }
    
    if not result.get('no_glass', True):
        # If glass is not visible but we have previous info, use that
        if not result.get('glass_visible', True) and prev_glass_info:
            frame_data.update({
                'water_percentage': -1,
                'left_point': None,
                'right_point': None,
                'glass_characteristics': prev_glass_info.get('glass_characteristics'),
                'appears_different': False,
                'drinking_event': result.get('drinking_event', False)
            })
        else:
            frame_data.update({
                'water_percentage': result.get('percentage', 0),
                'left_point': result.get('left'),
                'right_point': result.get('right'),
                'glass_characteristics': result.get('glass_characteristics'),
                'appears_different': result.get('appears_different', False),
                'drinking_event': result.get('drinking_event', False)
            })
    
    # Draw the analysis result on the frame
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        if not frame_data['has_glass']:
            cv2.putText(annotated_frame, "No glass detected", (50, 50), font, 1, (0, 0, 255), 2)
        elif frame_data['glass_visible']:
            # Draw water level line
            cv2.line(annotated_frame, 
                     (int(frame_data['left_point'][0]), int(frame_data['left_point'][1])), 
                     (int(frame_data['right_point'][0]), int(frame_data['right_point'][1])), 
                     (0, 0, 255), 2)
            
            # Add percentage text
            cv2.putText(annotated_frame, 
                        f"{frame_data['water_percentage']}% full", 
                        (int(frame_data['left_point'][0]), int(frame_data['left_point'][1] - 20)), 
                        font, 1, (0, 0, 255), 2)
    except Exception as e:
        print(f"Error annotating frame: {e}")
        return None
    
    # Save annotated frame
    output_filename = f"analyzed_{os.path.basename(frame_path)}"
    output_path = os.path.join('analyzed_frames', output_filename)
    cv2.imwrite(output_path, annotated_frame)
    
    return frame_data

def create_frame_collage(frames):
    """Create a 6x5 collage from a list of frames"""
    if not frames:
        return None
        
    # Get dimensions from first frame
    height, width = frames[0].shape[:2]
    
    # Create a blank canvas (5 rows, 6 columns)
    canvas = np.zeros((height * 5, width * 6, 3), dtype=np.uint8)
    
    # Place frames in grid
    for idx, frame in enumerate(frames[:30]):  # Only use first 30 frames
        row = idx // 6
        col = idx % 6
        y_start = row * height
        x_start = col * width
        canvas[y_start:y_start + height, x_start:x_start + width] = frame
    
    return canvas

def analyze_video_samples(video_path, dest, id):
    print("Starting video analysis...")
    
    # Delete existing files and folders
    if os.path.exists(f'{dest}/{id}/analyzed_frames'):
        shutil.rmtree(f'{dest}/{id}/analyzed_frames')
    if os.path.exists(f'{dest}/{id}/camera_frames'):
        shutil.rmtree(f'{dest}/{id}/camera_frames')
    if os.path.exists(f'{dest}/{id}/analysis_data.json'):
        os.remove(f'{dest}/{id}/analysis_data.json')
        
    # Create output directories
    os.makedirs(f'{dest}/{id}/camera_frames', exist_ok=True)
    os.makedirs(f'{dest}/{id}/analyzed_frames', exist_ok=True)
    
    # Read API key
    with open('custom-key.txt', 'r') as f:
        api_key = f.read().strip()

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = 30  # We'll process at 30fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Dictionary to store analysis data
    analysis_data = {'frames': {}}
    
    # Read video frame by frame
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save every frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_path = os.path.join('camera_frames', f'frame_{timestamp}.jpg')
        cv2.imwrite(frame_path, frame)
        
        # Analyze every 2 seconds (60 frames)
        if frame_count % 60 == 1:
            print(f"Processing frame {frame_count}")
            try:
                frame_data = analyze_and_save_frame(frame, frame_path, client, analysis_data)
                print(f"Frame data returned: {frame_data}")
                if frame_data:
                    analysis_data['frames'][timestamp] = frame_data
            except Exception as e:
                print(f"Error analyzing frame: {e}")
        
        frame_count += 1
        
    cap.release()
    
    # Save analysis data
    with open(f'{dest}/{id}/analysis_data.json', 'w') as f:
        json.dump(analysis_data, f, indent=4)
    
    print(f"\nAnalysis complete! Processed {len(analysis_data)} frames")
    print("You can now run create_smooth_animation.py to generate the animation")

if __name__ == "__main__":
    video_path = sys.argv[1]
    dest       = sys.argv[2]
    id         = sys.argv[3]
    analyze_video_samples(video_path, dest, id)
