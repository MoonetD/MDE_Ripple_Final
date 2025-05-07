import cv2
import numpy as np
import os
from datetime import datetime
import json
from PIL import Image, ImageDraw, ImageFont
import re
from collections import defaultdict
import logging

def get_frame_info(filename):
    """Extract timestamp from filename and return it along with the full path"""
    timestamp = re.search(r'(\d{8}_\d{6}_\d+)', filename).group(1)
    return timestamp, filename

def draw_water_level(frame, prev_data, next_data, factor):
    """Draw interpolated water level line and percentage on the frame"""
    if not prev_data['has_glass']:
        cv2.putText(frame, "No glass detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    if next_data and next_data['has_glass']:
        # Interpolate points
        prev_left = prev_data['left_point']
        prev_right = prev_data['right_point']
        next_left = next_data['left_point']
        next_right = next_data['right_point']
        
        left_x = int(prev_left[0] + (next_left[0] - prev_left[0]) * factor)
        left_y = int(prev_left[1] + (next_left[1] - prev_left[1]) * factor)
        right_x = int(prev_right[0] + (next_right[0] - prev_right[0]) * factor)
        right_y = int(prev_right[1] + (next_right[1] - prev_right[1]) * factor)
        
        # Interpolate percentage
        percentage = prev_data['water_percentage'] + \
                    (next_data['water_percentage'] - prev_data['water_percentage']) * factor
    else:
        # Use previous frame's data
        left_x, left_y = map(int, prev_data['left_point'])
        right_x, right_y = map(int, prev_data['right_point'])
        percentage = prev_data['water_percentage']
    
    # Draw water level line
    cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2)
    
    # Draw percentage text above the line
    text_y = min(left_y, right_y) - 30
    cv2.putText(frame, f"{int(percentage)}% full", 
                (left_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame

def interpolate_points(p1, p2, factor):
    """Interpolate between two points"""
    return [
        int(p1[0] + (p2[0] - p1[0]) * factor),
        int(p1[1] + (p2[1] - p1[1]) * factor)
    ]

def extract_text_from_image(image):
    """Extract text content from analyzed image using OCR or pattern matching"""
    # Convert to PIL Image for text extraction
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Convert to grayscale and threshold to isolate red text
    red_channel = image[:,:,2]
    _, thresh = cv2.threshold(red_channel, 200, 255, cv2.THRESH_BINARY)
    
    # If we find "No glass detected" text pattern
    if cv2.countNonZero(thresh) > 1000:  # Arbitrary threshold for text presence
        return "No glass detected"
    
    # Otherwise, try to find percentage pattern
    text = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(text[0]) > 0:
        # Look for pattern that matches "XX% full"
        text_area = np.mean([cv2.contourArea(c) for c in text[0]])
        if text_area > 100:  # Arbitrary threshold for text size
            return "percentage"  # Indicates water level is present
    return None

def main():
    # Create output directory for the interpolated frames
    os.makedirs('interpolated_frames', exist_ok=True)
    
    # Load analysis data from JSON
    try:
        with open('analysis_data.json', 'r') as f:
            analysis_data = json.load(f)
    except FileNotFoundError:
        print("No analysis data found! Please run live_water_analysis.py first.")
        return
    
    # Get all camera frames
    camera_frames = []
    for f in sorted(os.listdir('camera_frames')):
        if f.endswith('.jpg'):
            timestamp, path = get_frame_info(f)
            camera_frames.append((timestamp, path))
    
    if not camera_frames or not analysis_data['frames']:
        print("No frames found!")
        return
    
    # Find keyframes (frames with glass and water percentage)
    keyframes = []
    for timestamp, data in sorted(analysis_data['frames'].items()):
        if data['has_glass'] and 'water_percentage' in data:
            keyframes.append((timestamp, data))
    
    print(f"Found {len(camera_frames)} camera frames and {len(keyframes)} keyframes with water levels")
    
    if len(keyframes) < 2:
        print("Need at least 2 keyframes with water levels to create animation")
        return
    
    # Process each camera frame
    print("Creating interpolated frames...")
    frame_count = 0
    current_keyframe_idx = 0
    
    for i, (frame_timestamp, frame_path) in enumerate(camera_frames):
        # Read the frame
        frame = cv2.imread(os.path.join('camera_frames', frame_path))
        if frame is None:
            continue
        
        frame_dt = datetime.strptime(frame_timestamp, '%Y%m%d_%H%M%S_%f')
        
        # Find the appropriate keyframe pair for this frame
        while (current_keyframe_idx < len(keyframes) - 1 and 
               datetime.strptime(keyframes[current_keyframe_idx + 1][0], '%Y%m%d_%H%M%S_%f') <= frame_dt):
            current_keyframe_idx += 1
        
        # Skip if we're before the first keyframe
        if frame_dt < datetime.strptime(keyframes[0][0], '%Y%m%d_%H%M%S_%f'):
            continue
        
        # Get current keyframe pair
        prev_keyframe = keyframes[current_keyframe_idx]
        if current_keyframe_idx < len(keyframes) - 1:
            next_keyframe = keyframes[current_keyframe_idx + 1]
        else:
            # If we're after the last keyframe, just use the last keyframe's data
            output_path = os.path.join('interpolated_frames', f'interpolated_{i:06d}.jpg')
            cv2.imwrite(output_path, frame)
            continue
        
        # Calculate interpolation factor
        prev_time = datetime.strptime(prev_keyframe[0], '%Y%m%d_%H%M%S_%f')
        next_time = datetime.strptime(next_keyframe[0], '%Y%m%d_%H%M%S_%f')
        total_time = (next_time - prev_time).total_seconds()
        current_time = (frame_dt - prev_time).total_seconds()
        factor = current_time / total_time if total_time > 0 else 0
        
        # Get points from keyframes
        prev_data = prev_keyframe[1]
        next_data = next_keyframe[1]
        
        # Interpolate points
        prev_left = prev_data['left_point']
        prev_right = prev_data['right_point']
        next_left = next_data['left_point']
        next_right = next_data['right_point']
        
        left_x = int(prev_left[0] + (next_left[0] - prev_left[0]) * factor)
        left_y = int(prev_left[1] + (next_left[1] - prev_left[1]) * factor)
        right_x = int(prev_right[0] + (next_right[0] - prev_right[0]) * factor)
        right_y = int(prev_right[1] + (next_right[1] - prev_right[1]) * factor)
        
        # Interpolate percentage
        prev_pct = prev_data['water_percentage']
        next_pct = next_data['water_percentage']
        percentage = prev_pct + (next_pct - prev_pct) * factor
        
        # Draw water level line
        cv2.line(frame, (left_x, left_y), (right_x, right_y), (0, 0, 255), 2)
        
        # Draw text above the line
        text_y = min(left_y, right_y) - 30
        cv2.putText(frame, f"{int(percentage)}% full", (left_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Save the interpolated frame
        output_path = os.path.join('interpolated_frames', f'interpolated_{i:06d}.jpg')
        cv2.imwrite(output_path, frame)
        
        if i % 100 == 0:
            print(f"Processed {i} frames...")
    
    # Create video from interpolated frames
    print("\nCreating final video...")
    first_frame = cv2.imread(os.path.join('interpolated_frames', sorted(os.listdir('interpolated_frames'))[0]))
    height, width = first_frame.shape[:2]
    
    output_video = cv2.VideoWriter(
        'smooth_analysis.mp4',
        cv2.VideoWriter_fourcc(*'avc1'),
        30,  # fps
        (width, height)
    )
    
    for frame_name in sorted(os.listdir('interpolated_frames')):
        if frame_name.endswith('.jpg'):
            frame = cv2.imread(os.path.join('interpolated_frames', frame_name))
            if frame is not None:
                output_video.write(frame)
    
    output_video.release()
    print("\nDone! Video saved as 'smooth_analysis.mp4'")

if __name__ == "__main__":
    main()
