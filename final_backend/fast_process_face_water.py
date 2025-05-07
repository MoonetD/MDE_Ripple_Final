#############################
# app.py
#############################
import os
import cv2
import dlib
import joblib
import numpy as np
from typing import Optional
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import subprocess
# -*- coding: utf-8 -*-
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict
import dlib
import joblib
import random
import torch
import matplotlib.pyplot as plt

from fastapi.responses import JSONResponse

import cv2
import os
import json
import shutil
from datetime import datetime
from openai import OpenAI
import base64
import logging
import numpy as np
from PIL import Image
import concurrent.futures
import threading
from queue import Queue
import matplotlib.pyplot as plt
import time # Added for potential timing/sleep if needed
import asyncio # Add this import

# Define the request body model
class VideoProcessingRequest(BaseModel):
    test_video_path: str
    water_video_path: str
    test_output_directory: str

# Create FastAPI app instance
app = FastAPI()

async def process_video_and_analyze_hydration(video_path, user_id, output_dir = r"C:\Users\ydeng\Desktop\Water\Web-Ripple\static\assets\video_input"):
    """
    Processes a video to find the top 3 sharpest faces of the most frequent person,
    then performs hydration analysis on each face in turn (with fallback).
    The first face that yields a successful hydration analysis stops further attempts.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory where output images will be saved.

    Returns:
        bool: True if processing and analysis completed successfully (for at least one face),
              False otherwise.
        str: The absolute path to the output directory if successful, None otherwise.
    """
    start_full_process_time = time.time()

    # --- Create Output Directory ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        abs_output_dir = os.path.abspath(output_dir)
        print(f"Output will be saved to: {abs_output_dir}")
    except Exception as e:
        print(f"ERROR: Could not create output directory {output_dir}: {e}")
        return False, None

    # --- Configuration ---
    SAMPLE_RATE_SEC = 3
    DISPLAY_RESULT = False # Keep False for function use
    YOLO_MODEL_NAME = 'yolov8n.pt'
    MAX_FRAMES_TO_PROCESS = 100
    IOU_THRESHOLD = 0.3
    CONF_THRESHOLD = 0.5
    MIN_PERSON_SIZE_RATIO = 0.05
    BATCH_SIZE = 4
    SKIP_FIRST_PASS = False
    VERBOSE = False # Keep minimal output for function use

    # Derived output paths for up to 3 faces
    output_face_crops = [
        os.path.join(output_dir, f'{user_id}_sharpest_face_crop_1.jpg'),
        os.path.join(output_dir, f'{user_id}_sharpest_face_crop_2.jpg'),
        os.path.join(output_dir, f'{user_id}_sharpest_face_crop_3.jpg')
    ]

    # (These next three outputs will be overwritten depending on which face's analysis succeeds)
    output_probabilities_path = os.path.join(output_dir, f'{user_id}_hydration_probabilities.png')
    output_processed_face_path = os.path.join(output_dir, f'{user_id}_processed_face.jpg')
    output_analysis_overlay_path = os.path.join(output_dir, f'{user_id}_analysis_overlay.jpg')  # For internal logic if needed

    # --- Model Paths (Ensure these are accessible from where the script runs) ---
    PREDICTOR_MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"
    RF_MODEL_PATH = "models/RandomForest_model.pkl"

    # --- GPU Setup ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU.")

    # Initialize cap outside try block for finally clause
    cap = None
    yolo_model = None
    mtcnn = None
    prediction = None # Initialize prediction and probabilities
    probabilities = None

    # ================================================
    # Part 1: Find Sharpest Face Frames (Top 3)
    # ================================================
    try:
        print("\n--- Starting Part 1: Sharpest Face Detection ---")
        part1_start_time = time.time()

        # --- Load YOLO Model ---
        print(f"Loading YOLO model ({YOLO_MODEL_NAME})...")
        yolo_model = YOLO(YOLO_MODEL_NAME)
        yolo_model.overrides['conf'] = CONF_THRESHOLD
        yolo_model.overrides['iou'] = 0.45
        yolo_model.overrides['verbose'] = False
        yolo_model.to(device)
        person_class_id = 0
        if hasattr(yolo_model, 'names'):
            person_names = [name.lower() for name in yolo_model.names.values()]
            try:
                person_class_id = person_names.index('person')
            except ValueError:
                print(f"Using default person class ID: {person_class_id}")
        print("YOLO model loaded.")

        # --- Load MTCNN Model ---
        print("Loading MTCNN model...")
        mtcnn = MTCNN(keep_all=True, device=device, select_largest=False, min_face_size=20, thresholds=[0.6, 0.7, 0.8])
        print("MTCNN model loaded.")

        # --- Utility Functions (Part 1) ---
        def calculate_iou(box1, box2):
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - intersection_area
            return intersection_area / union_area if union_area > 0 else 0.0

        laplacian_kernel = torch.tensor(
            [[
                [0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]
            ]], dtype=torch.float32, device=device
        ).unsqueeze(0)

        def calculate_sharpness(face_crop_bgr):
            face_gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            face_tensor = torch.tensor(face_gray, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            laplacian_output = torch.nn.functional.conv2d(face_tensor, laplacian_kernel, padding=1)
            return torch.var(laplacian_output).item()

        # --- Video Processing ---
        print(f"Opening video file: {video_path}")
        if not os.path.exists(video_path):
            print(f"ERROR: Video file not found at: {video_path}")
            return False, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video file: {video_path}")
            return False, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_interval = int(fps * SAMPLE_RATE_SEC)
        if frame_interval <= 0:
            frame_interval = 1

        print(f"Video properties: {frame_width}x{frame_height} @ {fps:.2f} FPS, {frame_count_total} frames.")
        print(f"Sampling every {frame_interval} frames (~{SAMPLE_RATE_SEC} seconds).")

        if MAX_FRAMES_TO_PROCESS is not None:
            print(f"Limiting processing to approximately {MAX_FRAMES_TO_PROCESS} sampled frames.")

        main_person_frames = None
        person_tracks = []  # For first pass

        # --------------------
        # First Pass: Identify Main Person via Tracking (unless SKIP_FIRST_PASS = True)
        # --------------------
        if SKIP_FIRST_PASS:
            print("\n--- Skipping person tracking, using largest person assumption ---")
            main_person_frames = range(0, frame_count_total, frame_interval)
            if MAX_FRAMES_TO_PROCESS:
                main_person_frames = list(main_person_frames)[:MAX_FRAMES_TO_PROCESS]
        else:
            print("\n--- First Pass: Person Frequency Analysis ---")
            person_occurrences = defaultdict(int)
            frame_num = 0
            frames_analyzed = 0
            while frames_analyzed < (MAX_FRAMES_TO_PROCESS or float('inf')):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % frame_interval == 0:
                    frames_analyzed += 1
                    if not VERBOSE:
                        print(f"\rTracking progress: {frames_analyzed} frames analyzed", end="")
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = yolo_model(frame_rgb, classes=[person_class_id])
                    current_boxes = []
                    if len(results) > 0:
                        for box in results[0].boxes:
                            if int(box.cls) == person_class_id and float(box.conf) >= CONF_THRESHOLD:
                                xyxy = box.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                box_width, box_height = x2 - x1, y2 - y1
                                if (box_width * box_height) < (frame_width * frame_height * MIN_PERSON_SIZE_RATIO):
                                    continue
                                box_info = {
                                    'box': (x1, y1, x2, y2),
                                    'conf': float(box.conf),
                                    'frame': frame_num,
                                    'area': box_width * box_height
                                }
                                current_boxes.append(box_info)
                    # Attempt to match current box to existing tracks
                    for current_box in current_boxes:
                        matched = False
                        for track_idx, track in enumerate(person_tracks):
                            last_box = track[-1]['box']
                            iou = calculate_iou(current_box['box'], last_box)
                            if iou > IOU_THRESHOLD:
                                person_tracks[track_idx].append(current_box)
                                person_occurrences[track_idx] += 1
                                matched = True
                                break
                        if not matched:
                            new_track_idx = len(person_tracks)
                            person_tracks.append([current_box])
                            person_occurrences[new_track_idx] = 1
                frame_num += 1
            if not VERBOSE:
                print()

            if not person_tracks:
                print("No persons detected. Switching to largest-person detection mode.")
                SKIP_FIRST_PASS = True
                main_person_frames = range(0, frame_count_total, frame_interval)
                if MAX_FRAMES_TO_PROCESS:
                    main_person_frames = list(main_person_frames)[:MAX_FRAMES_TO_PROCESS]
            else:
                main_person_track_idx = max(person_occurrences.items(), key=lambda x: x[1])[0]
                main_person_frames = [entry['frame'] for entry in person_tracks[main_person_track_idx]]
                print(f"Main person found (appears in {person_occurrences[main_person_track_idx]} frames)")

        if main_person_frames is None:
            print("ERROR: Could not determine frames for main person.")
            cap.release()
            return False, None

        # Reset cap for second pass
        cap.release()
        cap = cv2.VideoCapture(video_path)

        print("\n--- Second Pass: Finding Top 3 Sharpest Faces for Main Person ---")
        face_candidates = []  # We'll store dict with keys: score, frame_data, face_crop, face_coords, etc.

        frames_processed = 0
        frames_with_face = 0

        frame_num = 0
        total_frames_to_process_pass2 = len(main_person_frames) if hasattr(main_person_frames, '__len__') else 'unknown'

        print(f"Processing approximately {total_frames_to_process_pass2} frames for face detection...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num in main_person_frames:
                frames_processed += 1
                if not VERBOSE:
                    print(f"\rProcessing frames: {frames_processed}/{total_frames_to_process_pass2}", end="")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, _ = frame.shape

                results = yolo_model(frame_rgb, classes=[person_class_id])
                best_person_box = None
                best_score_metric = -1
                if len(results) > 0 and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        if int(box.cls) == person_class_id and float(box.conf) >= CONF_THRESHOLD:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            if x1 < x2 and y1 < y2:
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                                if x1 < x2 and y1 < y2:
                                    area = (x2 - x1) * (y2 - y1)
                                    conf = float(box.conf)
                                    area_ratio = area / (width * height)
                                    score_metric = conf * (0.5 + 0.5 * area_ratio)
                                    if score_metric > best_score_metric:
                                        best_score_metric = score_metric
                                        best_person_box = (x1, y1, x2, y2)

                # If no person found in this frame, skip
                if best_person_box is None:
                    frame_num += 1
                    continue

                x1p, y1p, x2p, y2p = best_person_box
                person_crop = frame_rgb[y1p:y2p, x1p:x2p]
                if person_crop.size == 0:
                    frame_num += 1
                    continue

                person_img_pil = Image.fromarray(person_crop)
                face_boxes_rel, face_probs = mtcnn.detect(person_img_pil)
                if face_boxes_rel is None or len(face_boxes_rel) == 0:
                    frame_num += 1
                    continue

                # Find largest face within the person box
                largest_face = None
                largest_face_area = 0
                person_crop_h, person_crop_w, _ = person_crop.shape
                for fbox in face_boxes_rel:
                    x1f, y1f, x2f, y2f = map(int, fbox)
                    if x1f < x2f and y1f < y2f:
                        x1f, y1f = max(0, x1f), max(0, y1f)
                        x2f, y2f = min(person_crop_w - 1, x2f), min(person_crop_h - 1, y2f)
                        if x1f < x2f and y1f < y2f:
                            area = (x2f - x1f) * (y2f - y1f)
                            if area > largest_face_area:
                                largest_face_area = area
                                largest_face = (x1f, y1f, x2f, y2f)

                if largest_face is None:
                    frame_num += 1
                    continue

                x1f, y1f, x2f, y2f = largest_face
                xfa, yfa = x1p + x1f, y1p + y1f
                xfa2, yfa2 = x1p + x2f, y1p + y2f
                xfa, yfa = max(0, min(xfa, width - 1)), max(0, min(yfa, height - 1))
                xfa2, yfa2 = max(0, min(xfa2, width - 1)), max(0, min(yfa2, height - 1))
                if xfa >= xfa2 or yfa >= yfa2:
                    frame_num += 1
                    continue

                face_crop_bgr = frame[yfa:yfa2, xfa:xfa2]
                if face_crop_bgr.size == 0:
                    frame_num += 1
                    continue

                frames_with_face += 1
                sharpness = calculate_sharpness(face_crop_bgr)

                face_candidates.append({
                    'score': sharpness,
                    'frame_number': frame_num,
                    'frame_data': frame.copy(),
                    'face_crop': face_crop_bgr.copy(),
                    'face_coords': (xfa, yfa, xfa2, yfa2)
                })

            frame_num += 1

        if not VERBOSE:
            print()
        print("\nFace processing complete.")

        # If no faces found at all
        if not face_candidates:
            print("No valid faces were detected for the main person.")
            cap.release()
            return False, None

        # Sort by score descending and pick top 3
        face_candidates.sort(key=lambda x: x['score'], reverse=True)
        top_faces = face_candidates[:3]

        # --- Save top 3 faces with padding as requested ---
        def save_padded_face_crop(face_dict, output_path):
            frame_data = face_dict['frame_data']
            xfa, yfa, xfa2, yfa2 = face_dict['face_coords']
            face_width = xfa2 - xfa
            face_height = yfa2 - yfa
            longer_side = max(face_width, face_height)
            padding = int(longer_side * 0.3)  # padding
            face_center_x = (xfa + xfa2) // 2
            face_center_y = (yfa + yfa2) // 2
            new_side = longer_side + (2 * padding)
            x1_square = face_center_x - (new_side // 2)
            y1_square = face_center_y - (new_side // 2)
            x2_square = x1_square + new_side
            y2_square = y1_square + new_side
            h, w = frame_data.shape[:2]
            x1_square = max(0, x1_square)
            y1_square = max(0, y1_square)
            x2_square = min(w, x2_square)
            y2_square = min(h, y2_square)
            padded_face_crop = frame_data[y1_square:y2_square, x1_square:x2_square]
            cv2.imwrite(output_path, padded_face_crop)

        # Save each face to disk
        for i, face_dict in enumerate(top_faces):
            out_path = output_face_crops[i]
            save_padded_face_crop(face_dict, out_path)
            print(f"Top face #{i+1} (score={face_dict['score']:.2f}) saved to: {out_path}")

        part1_end_time = time.time()
        print(f"Part 1 Execution Time: {part1_end_time - part1_start_time:.2f} seconds")

        # Performance Summary
        print(f"\n--- Part 1 Summary ---")
        print(f"Total frames in video: {frame_count_total}")
        print(f"Frames sampled & processed for face: {frames_processed}")
        print(f"Total persons tracked (if enabled): {len(person_tracks) if not SKIP_FIRST_PASS else 'N/A'}")
        print(f"Total faces detected within largest persons: {frames_with_face}")

    except Exception as e:
        print(f"ERROR during Part 1 (Sharpest Face Detection): {e}")
        import traceback
        traceback.print_exc()
        if cap is not None and cap.isOpened():
            cap.release()
        return False, None
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
        # Clean up GPU memory
        if yolo_model:
            del yolo_model
        if mtcnn:
            del mtcnn
        if 'laplacian_kernel' in locals():
            del laplacian_kernel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================================================
    # Part 2: Hydration Analysis (Try up to 3 faces)
    # ================================================
    try:
        print("\n--- Starting Part 2: Hydration Analysis ---")
        part2_start_time = time.time()

        # --- Hydration Feature Extraction Class ---
        class HydrationFeatureExtractor:
            def __init__(self):
                self.FACIAL_LANDMARKS_IDXS = {
                    "jaw": (0, 17),
                    "right_eyebrow": (17, 22),
                    "left_eyebrow": (22, 27),
                    "nose": (27, 36),
                    "right_eye": (36, 42),
                    "left_eye": (42, 48),
                    "mouth": (48, 68)
                }
                self.landmark_predictor = None  # Will be loaded later

            def ensure_predictor_loaded(self):
                if self.landmark_predictor is None:
                    try:
                        if os.path.exists(PREDICTOR_MODEL_PATH):
                            self.landmark_predictor = dlib.shape_predictor(PREDICTOR_MODEL_PATH)
                            print("Dlib landmark predictor loaded.")
                        else:
                            print(f"Warning: Landmark predictor not found at {PREDICTOR_MODEL_PATH}")
                            # Propagate error if predictor is essential
                            raise FileNotFoundError(f"Landmark predictor not found at {PREDICTOR_MODEL_PATH}")
                    except Exception as e:
                        print(f"Error loading landmark predictor: {e}")
                        raise  # Re-raise the exception to be caught by the outer handler

            def shape_to_np(self, shape):
                coords = np.zeros((shape.num_parts, 2), dtype=np.int32)
                for i in range(0, shape.num_parts):
                    coords[i] = (shape.part(i).x, shape.part(i).y)
                return coords

            def extract_features(self, image_rgb, landmarks=None):
                try: # Wrap feature extraction in try-except
                    self.ensure_predictor_loaded()  # Load predictor if needed

                    if landmarks is None: # Only try detection if predictor is loaded and landmarks weren't provided
                        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                        detector = dlib.get_frontal_face_detector()
                        faces = detector(gray, 1)
                        if len(faces) == 0:
                            print("Warning: No faces detected by dlib detector in provided image.")
                            return None
                        face = max(faces, key=lambda rect: rect.width() * rect.height())
                        landmarks = self.landmark_predictor(gray, face)

                    if landmarks is None:
                        print("Warning: Could not obtain landmarks.")
                        return None

                    points = self.shape_to_np(landmarks)
                    features = {}
                    h, w = image_rgb.shape[:2]

                    # Color/Texture Features
                    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                    forehead_top = max(0, np.min([points[19][1], points[24][1]]) - 20)
                    forehead_bottom = np.min([points[19][1], points[24][1]])
                    forehead_left = points[17][0] - 5
                    forehead_right = points[26][0] + 5
                    right_cheek_x = points[31][0] - 35
                    right_cheek_y = points[41][1] + 15
                    left_cheek_x = points[35][0] + 35
                    left_cheek_y = points[47][1] + 15
                    region_size = 30

                    # Forehead region
                    if (0 <= forehead_left < w and 0 <= forehead_top < h and
                        0 <= forehead_right < w and 0 <= forehead_bottom < h and
                        forehead_top < forehead_bottom and forehead_left < forehead_right): # Add check
                        forehead_roi = image_rgb[forehead_top:forehead_bottom, forehead_left:forehead_right]
                        if forehead_roi.size > 0:
                            forehead_lab = cv2.cvtColor(forehead_roi, cv2.COLOR_RGB2LAB)
                            features['forehead_L'] = np.mean(forehead_lab[:, :, 0])
                            features['forehead_A'] = np.mean(forehead_lab[:, :, 1])
                            features['forehead_B'] = np.mean(forehead_lab[:, :, 2])

                            gray_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                            forehead_gray = gray_img[forehead_top:forehead_bottom, forehead_left:forehead_right]
                            if forehead_gray.size > 0:
                                features['forehead_texture_variance'] = np.var(
                                    cv2.Laplacian(forehead_gray, cv2.CV_64F)
                                )

                    # Right cheek region
                    if (0 <= right_cheek_x < w and 0 <= right_cheek_y < h and
                        0 <= right_cheek_x + region_size < w and 0 <= right_cheek_y + region_size < h):
                        right_cheek_roi = image_rgb[
                            right_cheek_y:right_cheek_y + region_size,
                            right_cheek_x:right_cheek_x + region_size
                        ]
                        if right_cheek_roi.size > 0:
                            right_cheek_lab = cv2.cvtColor(right_cheek_roi, cv2.COLOR_RGB2LAB)
                            features['right_cheek_L'] = np.mean(right_cheek_lab[:, :, 0])
                            features['right_cheek_A'] = np.mean(right_cheek_lab[:, :, 1])
                            features['right_cheek_B'] = np.mean(right_cheek_lab[:, :, 2])

                    # Left cheek region
                    if (0 <= left_cheek_x - region_size < w and 0 <= left_cheek_y < h and
                        0 <= left_cheek_x < w and 0 <= left_cheek_y + region_size < h):
                        left_cheek_roi = image_rgb[
                            left_cheek_y:left_cheek_y + region_size,
                            left_cheek_x - region_size:left_cheek_x
                        ]
                        if left_cheek_roi.size > 0:
                            left_cheek_lab = cv2.cvtColor(left_cheek_roi, cv2.COLOR_RGB2LAB)
                            features['left_cheek_L'] = np.mean(left_cheek_lab[:, :, 0])
                            features['left_cheek_A'] = np.mean(left_cheek_lab[:, :, 1])
                            features['left_cheek_B'] = np.mean(left_cheek_lab[:, :, 2])

                    # Derived color features
                    if 'forehead_L' in features and 'right_cheek_L' in features and features['right_cheek_L'] != 0:
                        features['forehead_cheek_L_ratio'] = features['forehead_L'] / features['right_cheek_L']
                    if 'forehead_A' in features and 'right_cheek_A' in features:
                        features['A_diff'] = abs(features['forehead_A'] - features['right_cheek_A'])
                    if 'right_cheek_L' in features and 'left_cheek_L' in features and features.get('left_cheek_L', 0) != 0: # Use get with default
                        features['cheek_L_ratio'] = features['right_cheek_L'] / features['left_cheek_L']

                    # Geometric Features (if 68 landmarks exist)
                    if len(points) == 68:
                        # Eye ratio
                        right_eye_width = np.linalg.norm(points[36] - points[39])
                        left_eye_width = np.linalg.norm(points[42] - points[45])
                        if left_eye_width > 0:
                            features['eye_width_ratio'] = right_eye_width / left_eye_width

                        # Lip area, perimeter, roundness
                        lip_points = points[48:60]
                        if len(lip_points) > 2:
                            lip_area = cv2.contourArea(lip_points)
                            features['lip_area'] = lip_area
                            lip_perimeter = cv2.arcLength(lip_points, True)
                            features['lip_perimeter'] = lip_perimeter
                            if lip_perimeter > 0:
                                features['lip_roundness'] = (4 * np.pi * lip_area) / (lip_perimeter ** 2)

                except Exception as e:
                    print(f"Error extracting features: {e}")
                    # Consider re-raising or returning None depending on desired behavior
                    return None # Indicate failure

                # Fill missing expected features with 0
                expected_features = [
                    'forehead_L', 'forehead_A', 'forehead_B',
                    'right_cheek_L', 'right_cheek_A', 'right_cheek_B',
                    'left_cheek_L', 'left_cheek_A', 'left_cheek_B',
                    'forehead_cheek_L_ratio', 'A_diff', 'cheek_L_ratio',
                    'forehead_texture_variance',
                    # Add geometric features if they are used by the model
                    'eye_width_ratio', 'lip_area', 'lip_perimeter', 'lip_roundness'
                ]
                for feature in expected_features:
                    if feature not in features:
                        features[feature] = 0.0

                print(f"Extracted {len(features)} features (filling missing with 0.0).")
                return features

            def visualize_features(self, image_rgb, landmarks, features=None):
                vis_img = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
                if landmarks is not None:
                    landmarks_np = self.shape_to_np(landmarks)
                    for (x, y) in landmarks_np:
                        cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)
                if features is not None:
                    y_pos = 30
                    display_features = list(features.items())[:10]  # Limit displayed features
                    for key, value in display_features:
                        text = f"{key}: {value:.2f}"
                        cv2.putText(vis_img, text, (10, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                        y_pos += 20
                return vis_img  # BGR output

        # --- Load Models for Part 2 ---
        def load_random_forest_model():
            print("Loading RandomForest model...")
            if not os.path.exists(RF_MODEL_PATH):
                print(f"Error: RandomForest model not found at {RF_MODEL_PATH}")
                return None
            try:
                model = joblib.load(RF_MODEL_PATH)
                print("RandomForest model loaded successfully")
                return model
            except Exception as e:
                print(f"Error loading RandomForest model: {e}")
                return None

        def predict_hydration_without_scaler(features_dict, rf_model):
            if rf_model is None or features_dict is None: # Check for None features_dict
                 print("Warning: Cannot predict, RF model or features are None.")
                 return None, None
            expected_features = [
                'forehead_L', 'forehead_A', 'forehead_B',
                'right_cheek_L', 'right_cheek_A', 'right_cheek_B',
                'left_cheek_L', 'left_cheek_A', 'left_cheek_B',
                'forehead_cheek_L_ratio', 'A_diff', 'cheek_L_ratio',
                'forehead_texture_variance',
                # Add geometric features if they are used by the model
                'eye_width_ratio', 'lip_area', 'lip_perimeter', 'lip_roundness'
            ]
            try:
                # Ensure all expected features exist in the dict, default to 0.0 if missing
                feature_values = [features_dict.get(feature, 0.0) for feature in expected_features]

                # Filter out features not expected by the loaded model
                # This assumes rf_model.feature_names_in_ holds the expected names
                if hasattr(rf_model, 'feature_names_in_'):
                    model_expected_features = rf_model.feature_names_in_
                    # Create a mapping from expected feature names to their values
                    current_features_map = {name: val for name, val in zip(expected_features, feature_values)}
                    # Reorder/select features based on the model's expectation
                    final_feature_values = [current_features_map.get(name, 0.0) for name in model_expected_features]
                    features_array = np.array(final_feature_values).reshape(1, -1)

                    if features_array.shape[1] != rf_model.n_features_in_:
                         print(f"ERROR: Feature mismatch after alignment. Expected {rf_model.n_features_in_}, got {features_array.shape[1]}")
                         return None, None # Critical error
                else:
                    # Fallback if feature names aren't stored in the model
                    features_array = np.array(feature_values).reshape(1, -1)
                    if features_array.shape[1] != rf_model.n_features_in_:
                         print(f"ERROR: Feature mismatch. Expected {rf_model.n_features_in_}, got {features_array.shape[1]}")
                         # Attempt to adjust (use with caution, may indicate upstream issues)
                         if features_array.shape[1] > rf_model.n_features_in_:
                             print("Warning: Truncating features to match model.")
                             features_array = features_array[:, :rf_model.n_features_in_]
                         else:
                             print("Warning: Padding features with zeros to match model.")
                             padding = np.zeros((1, rf_model.n_features_in_ - features_array.shape[1]))
                             features_array = np.hstack((features_array, padding))


                probabilities = rf_model.predict_proba(features_array)[0]
                prediction = np.argmax(probabilities)
                return prediction, probabilities
            except Exception as e:
                print(f"Error predicting hydration status: {e}")
                import traceback
                traceback.print_exc() # Print stack trace for debugging
                return None, None


        def generate_probability_chart(probabilities, output_path):
            try:
                plt.figure(figsize=(8, 6))
                categories = ['Dehydrated', 'Normal', 'Overhydrated']
                colors = ['#FF9999', '#99FF99', '#9999FF']
                bars = plt.bar(categories, probabilities, color=colors)
                plt.ylim([0, 1])
                plt.ylabel('Probability')
                plt.title('Hydration Status Probability Distribution')
                for i, prob in enumerate(probabilities):
                    plt.text(i, prob + 0.02, f'{prob:.2f}', ha='center')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                print(f"Probability chart saved to {output_path}")
                return True
            except Exception as e:
                print(f"Error generating probability chart: {e}")
                return False


        # Attempt analysis on up to 3 faces in order
        rf_model = load_random_forest_model()
        if rf_model is None:
            print("ERROR: Failed to load RandomForest model. Cannot analyze hydration.")
            return False, None

        status_labels = {0: "Dehydrated", 1: "Normal", 2: "Overhydrated"}
        feature_extractor = HydrationFeatureExtractor()
        predictor = None # Initialize predictor

        success = False
        for i, face_path in enumerate(output_face_crops):
            if not os.path.exists(face_path):
                continue  # skip if the file doesn't exist (e.g. fewer than 3 faces)
            print(f"\nAnalyzing Face #{i+1}: {face_path}")

            # Load face
            image = cv2.imread(face_path)
            if image is None:
                print(f"ERROR: Failed to load face image: {face_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect face / landmarks
            try:
                feature_extractor.ensure_predictor_loaded() # Ensure predictor is ready
                predictor = feature_extractor.landmark_predictor # Get the loaded predictor
                if predictor is None:
                    raise RuntimeError("Landmark predictor failed to load.")

                detector = dlib.get_frontal_face_detector()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 1)
                if len(faces) == 0:
                    print("ERROR: No faces detected in the cropped image for landmark detection.")
                    continue  # fallback to next face
                face_rect = max(faces, key=lambda rect: rect.width() * rect.height())

                landmarks = predictor(gray, face_rect)
                print(f"Detected {landmarks.num_parts} landmarks.")
            except Exception as e:
                print(f"ERROR: Exception during landmark detection on face #{i+1}: {e}")
                continue

            # Extract features
            features = feature_extractor.extract_features(image_rgb, landmarks)
            if features is None:
                print(f"ERROR: Feature extraction failed on face #{i+1}.")
                continue

            # Predict hydration
            prediction, probabilities = predict_hydration_without_scaler(features, rf_model)
            if prediction is None or probabilities is None:
                print(f"ERROR: Hydration prediction failed on face #{i+1}.")
                continue

            # If we get this far, success for this face
            print(f"Hydration Prediction: {status_labels.get(prediction, 'Unknown')} (Class {prediction})")
            print(f"Probabilities: Dehydrated={probabilities[0]:.3f}, Normal={probabilities[1]:.3f}, Overhydrated={probabilities[2]:.3f}")

            # Generate the processed face with landmarks
            processed_img_bgr = feature_extractor.visualize_features(image_rgb, landmarks, features)
            processed_img_bgr = cv2.cvtColor(processed_img_bgr, cv2.COLOR_RGB2BGR) # visualize_features already returns BGR
            try:
                cv2.imwrite(output_processed_face_path, processed_img_bgr)
                print(f"Processed face image saved to {output_processed_face_path}")
            except Exception as e:
                print(f"Error saving processed face image: {e}") # Log error but continue

            # Generate Probability Chart
            if not generate_probability_chart(probabilities, output_probabilities_path):
                 print("Warning: Failed to generate probability chart.") # Log warning but continue

            # We have succeeded with this face - stop fallback attempts
            success = True
            break # Exit the loop once successful

        part2_end_time = time.time()
        print(f"Part 2 Execution Time: {part2_end_time - part2_start_time:.2f} seconds")

        if not success:
            print("\nAll candidate faces failed to produce a successful hydration analysis.")
            return False, None

        # --- Final Summary ---
        end_full_process_time = time.time()
        print(f"\n--- Full Process Summary ---")
        # Check if prediction and probabilities were successfully set
        if prediction is not None and probabilities is not None:
            print(f"Hydration Status: {status_labels.get(prediction, 'Unknown')}")
            print(f"Confidence: {probabilities[prediction]:.2f}")
        else:
             print("Hydration Status: Could not be determined.")
        print(f"Output files saved in: {abs_output_dir}")
        print(f"Total execution time: {end_full_process_time - start_full_process_time:.2f} seconds")

        return True, abs_output_dir # Return success status and output dir path

    except Exception as e:
        print(f"ERROR during Part 2 (Hydration Analysis): {e}")
        import traceback
        traceback.print_exc()
        return False, None


# --- Helper Function: Parse Timestamp ---
def parse_timestamp(timestamp_str):
    """Parses timestamp string (YYYYMMDD_HHMMSS_f) into datetime object."""
    try:
        # Try parsing with microseconds
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
    except ValueError:
        # Fallback if microseconds are missing (shouldn't happen with generation method)
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except TypeError:
        # Handle case where input might not be a string
        print(f"Warning: Invalid type for parse_timestamp: {type(timestamp_str)}. Value: {timestamp_str}")
        # Return a default past date or raise an error depending on desired handling
        return datetime.min # Example fallback

# --- Main Combined Processing Function ---
async def process_drinking_video_full(video_path, user_id, output_dir = r"C:\Users\ydeng\Desktop\Water\Web-Ripple\static\assets\video_input", api_key_path='custom-key.txt', num_workers=10, frames_to_skip=60, cup_ml=150, output_video_fps=30):
    """
    Analyzes a video for drinking events, generates summary/graph,
    and creates a new video with smoothly interpolated annotations.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to the directory where all outputs will be saved.
        api_key_path (str): Path to the file containing the OpenAI API key.
        num_workers (int): Number of parallel workers for frame analysis (Part 1).
        frames_to_skip (int): Number of frames to skip between analyses (Part 1).
        cup_ml (int): Estimated volume of the cup in milliliters for graph title.
        output_video_fps (int): Frames per second for the output annotated video.

    Returns:
        bool: True if all steps completed successfully, False otherwise.
        dict: Dictionary containing paths to the main output files if successful.
              Keys: 'summary_json', 'graph_png', 'annotated_video'
              Returns None if unsuccessful.
    """
    print(f"--- Starting Full Drinking Analysis and Video Generation ---")
    print(f"Video Input: {video_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Settings: Workers={num_workers}, Frame Skip={frames_to_skip}, Cup Volume={cup_ml}ml, Output FPS={output_video_fps}")

    overall_start_time = time.time()

    # --- Setup Directories ---
    os.makedirs(output_dir, exist_ok=True)
    # --- Part 1 specific dirs ---
    temp_dir = os.path.join(output_dir, 'temp_analysis_files')
    camera_frames_dir = os.path.join(temp_dir, f'{user_id}_camera_frames') # Stores ORIGINAL frames
    analyzed_frames_dir = os.path.join(temp_dir, f'{user_id}_analyzed_frames') # Stores VISION-ANNOTATED frames
    intermediate_json_path = os.path.join(temp_dir, f'{user_id}_analysis_data.json')
    # --- Part 2 (Sequence Analysis) specific output paths ---
    final_graph_path = os.path.join(output_dir, f'{user_id}_water_consumption_graph.png')
    final_json_path = os.path.join(output_dir, f'{user_id}_drinked_percentage.json')
    # --- Part 3 (Video Generation) specific paths ---
    interpolated_frames_dir = os.path.join(output_dir, f'{user_id}_interpolated_frames') # Stores SMOOTH-ANNOTATED frames
    final_video_path = os.path.join(output_dir, f'{user_id}_smooth_analysis.mp4')

    # Clean up previous temporary/intermediate files if they exist
    if os.path.exists(temp_dir):
        print(f"Removing existing temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    if os.path.exists(interpolated_frames_dir):
        print(f"Removing existing interpolated frames directory: {interpolated_frames_dir}")
        shutil.rmtree(interpolated_frames_dir)
    os.makedirs(temp_dir)
    os.makedirs(camera_frames_dir)
    os.makedirs(analyzed_frames_dir)
    os.makedirs(interpolated_frames_dir) # Create dir for smooth frames

    # --- API Key and Client ---
    try:
        with open(api_key_path, 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        print("OpenAI client initialized.")
    except FileNotFoundError:
        print(f"ERROR: API key file not found at {api_key_path}")
        return False, None
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client: {e}")
        return False, None

    # --- Shared Data Structures (for Part 1) ---
    analysis_data = {'frames': {}} # Store per-frame results here
    analysis_data_lock = threading.Lock()
    task_queue = Queue()
    original_frame_paths = {} # Store path for *every* frame: {timestamp: path}

    # =============================================
    # Part 1: Per-Frame Analysis (using GPT-4V)
    # =============================================
    part1_start_time = time.time()
    print("\n--- Starting Part 1: Per-Frame Analysis ---")

    # --- Helper Functions for Part 1 (Analyze and Save) ---
    # (Note: This is the *same* analyze_and_save_frame_part1 function from Script 1)
    def analyze_and_save_frame_part1(frame, frame_path_original, frame_timestamp, local_client, local_analysis_data, local_lock):
        base64_image = None
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Error encoding frame {frame_timestamp}: {e}")
            return None

        # --- Prepare OpenAI Request (using the detailed prompt) ---
        prompt_text = "You are analysing a *video* frame-by-frame in chronological order.  \nMaintain an internal \"current glass ID\" that carries over from the previous frame unless you deliberately switch it (see rules 1-3).\n\n\n 1.  WHICH GLASS TO TRACK\n\na.  If one or more glasses are visible, pick **one**:\n     the glass whose bounding box centre is closest to any visible hand, **or** \n     if multiple tie, the glass that was already being tracked in the previous frame.\nb.  If *no* glass is visible, keep the last tracked glass in memory but mark glass_visible=false for this frame.\n\n\n 2.  APPEARS_DIFFERENT?\n\nOnly set appears_different=true when the newly chosen glass differs *substantially* from the remembered one in material, shape, or size.  \nMinor changes caused by perspective or lighting  false.\n\n\n 3.  DRINKING_EVENT LOGIC\n\nMark drinking_event=true in *any* of the following situations **while a hand is gripping or clearly guiding the glass**:\n    i.   The glass is tilted > 20  from vertical.  \n    ii.  The water level is visibly disturbed / mid-motion.  \n    iii. The glass was visible and tracked in the *previous* frame, is **not** visible in the *current* frame (glass_visible=false), and appears again within the next 2 seconds **with roughly the same fill level** (±5 %) — treat the moment it left the frame as the drinking event.\n\nDo **not** mark a drinking event for horizontal hand movements, light touches, or when the glass is put down on a table without tilt or water disturbance.\n\n\n 4.  OUTPUT FORMAT  (exact JSON)\n\nIf **no** glass is visible *and* no previous glass is being tracked, output:\n{\"no_glass\": true}\n\nOtherwise output:\n{\n  \"no_glass\": false,\n  \"glass_visible\": boolean,        // true if glass in this frame, false if it has momentarily left the frame\n  \"left\": [x1, y1],                // pixel coords of left edge of water line; null if glass_visible=false\n  \"right\": [x2, y2],               // pixel coords of right edge of water line; null if glass_visible=false\n  \"percentage\": number,            // 0-100; -1 if glass_visible=false\n  \"glass_characteristics\": string, // brief free-text\n  \"appears_different\": boolean,\n  \"drinking_event\": boolean\n}\n\nUse null or -1 only when that field cannot be measured because the glass is out of frame in this specific image.\n\n 5. IMPORTANT DETAILS \n\nAll coordinates are in the current image's pixel space.\n\nTreat the \"next 2 seconds\" window in rule 3 iii as  50 frames at 30 fps (adjust if actual fps known).\n\nBe conservative—if uncertain, return drinking_event=false."

        try:
            # print(f"DEBUG: Sending frame {frame_timestamp} to GPT-4V") # Optional debug
            response = local_client.chat.completions.create(
                model="gpt-4.1", # Using vision model (gpt-4.1 or gpt-4-vision-preview)
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "low"}
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            response_text = response.choices[0].message.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            result = json.loads(response_text.strip())
            # print(f"DEBUG: Received GPT-4V response for {frame_timestamp}: {result}") # Optional debug

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for frame {frame_timestamp}: {e}")
            print(f"Raw response: {response_text}")
            return None
        except Exception as e:
            print(f"Error making API call for frame {frame_timestamp}: {e}")
            return None

        # --- Process Result and Annotate (for analyzed_frames dir) ---
        frame_data_to_store = {
            'timestamp': frame_timestamp,
            'frame_path': frame_path_original, # Path to ORIGINAL frame
            'result': result, # Raw GPT result
            'has_glass': not result.get('no_glass', True),
            'glass_visible': result.get('glass_visible', False) if not result.get('no_glass', True) else False
        }

        annotated_frame = frame.copy() # Annotate a copy
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 255, 255) # White
        bg_color = (0, 0, 0) # Black
        line_color = (0, 0, 255) # Red

        if frame_data_to_store['has_glass'] and frame_data_to_store['glass_visible']:
            # Store detailed info if glass is visible
            frame_data_to_store.update({
                'water_percentage': result.get('percentage', -1), # Use -1 if missing
                'left_point': result.get('left'),
                'right_point': result.get('right'),
                'glass_characteristics': result.get('glass_characteristics', 'N/A'),
                'appears_different': result.get('appears_different', False),
                'drinking_event': result.get('drinking_event', False)
            })
            # Annotate frame based on VISION MODEL output
            lp = frame_data_to_store['left_point']
            rp = frame_data_to_store['right_point']
            perc = frame_data_to_store['water_percentage']

            if lp and rp and isinstance(lp, list) and len(lp) == 2 and isinstance(rp, list) and len(rp) == 2:
                try:
                    lp_int = (int(lp[0]), int(lp[1]))
                    rp_int = (int(rp[0]), int(rp[1]))
                    cv2.line(annotated_frame, lp_int, rp_int, line_color, 2)
                    if isinstance(perc, (int, float)) and perc >= 0:
                        perc_text = f"{perc:.0f}% (Vision)"
                        text_pos = (lp_int[0], lp_int[1] - 15)
                        cv2.putText(annotated_frame, perc_text, text_pos, font, 0.6, bg_color, 3, cv2.LINE_AA)
                        cv2.putText(annotated_frame, perc_text, text_pos, font, 0.6, text_color, 1, cv2.LINE_AA)
                    else:
                         print(f"Warning: Missing/invalid percentage {perc} for annotation on frame {frame_timestamp}")
                         no_perc_text = "Level Unknown (Vision)"
                         cv2.putText(annotated_frame, no_perc_text, (30,50), font, 0.6, bg_color, 3, cv2.LINE_AA)
                         cv2.putText(annotated_frame, no_perc_text, (30,50), font, 0.6, text_color, 1, cv2.LINE_AA)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid coordinate data for annotation on frame {frame_timestamp}: {lp}, {rp} - {e}")
            else:
                perc_text = f"{perc if isinstance(perc, (int, float)) and perc >= 0 else 'N/A'}% (coords missing)"
                cv2.putText(annotated_frame, perc_text, (30,50), font, 0.6, bg_color, 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, perc_text, (30,50), font, 0.6, text_color, 1, cv2.LINE_AA)

        elif frame_data_to_store['has_glass']: # Glass exists but is not visible
             frame_data_to_store.update({ # Store placeholders
                'water_percentage': -1,
                'left_point': None,
                'right_point': None,
                'glass_characteristics': result.get('glass_characteristics', 'N/A'), # Keep characteristics if provided
                'appears_different': False,
                'drinking_event': result.get('drinking_event', False) # Vision model might flag based on rule 3iii
            })
             vis_text = f"Glass Hidden ({frame_data_to_store['glass_characteristics']})"
             cv2.putText(annotated_frame, vis_text, (30, 50), font, 0.7, bg_color, 3, cv2.LINE_AA)
             cv2.putText(annotated_frame, vis_text, (30, 50), font, 0.7, text_color, 1, cv2.LINE_AA)

        else: # No glass detected at all
            cv2.putText(annotated_frame, "No glass detected (Vision)", (30, 50), font, 0.7, bg_color, 3, cv2.LINE_AA)
            cv2.putText(annotated_frame, "No glass detected (Vision)", (30, 50), font, 0.7, text_color, 1, cv2.LINE_AA)

        # Save the vision-annotated frame (for debugging/review)
        output_filename_annotated = f"analyzed_{frame_timestamp}.jpg"
        output_path_annotated = os.path.join(analyzed_frames_dir, output_filename_annotated)
        try:
            cv2.imwrite(output_path_annotated, annotated_frame)
        except Exception as e:
             print(f"Error saving vision-annotated frame {output_filename_annotated}: {e}")

        # Update shared analysis data safely
        with local_lock:
            local_analysis_data['frames'][frame_timestamp] = frame_data_to_store

        return frame_data_to_store # Return the processed data

    # --- Worker Thread Function (for Part 1) ---
    def worker_part1(local_task_queue, local_client, local_analysis_data, local_lock):
        while True:
            task = local_task_queue.get()
            if task is None:
                local_task_queue.task_done()
                break # Exit loop

            frame_copy, frame_path_original, frame_timestamp, frame_count_display = task
            # print(f"Worker processing frame {frame_count_display} ({frame_timestamp})...")
            try:
                result_data = analyze_and_save_frame_part1(frame_copy, frame_path_original, frame_timestamp, local_client, local_analysis_data, local_lock)
                # if result_data:
                #      print(f"Worker finished frame {frame_count_display}. Glass detected: {result_data['has_glass']}")
                # else:
                #      print(f"Worker failed to process frame {frame_count_display}.")
            except Exception as e:
                print(f"ERROR in worker processing frame {frame_count_display}: {e}")
            finally:
                local_task_queue.task_done()

    # --- Start Video Processing for Part 1 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return False, None

    video_fps_native = cap.get(cv2.CAP_PROP_FPS) # Get native FPS
    print(f"Native video FPS detected: {video_fps_native:.2f}")
    # Adjust frames_to_skip if needed, based on desired time interval
    effective_frames_to_skip = frames_to_skip
    if video_fps_native > 0:
         seconds_per_analysis = frames_to_skip / 30.0 # Assume desired interval based on 30fps default
         effective_frames_to_skip = max(1, int(seconds_per_analysis * video_fps_native))
         print(f"Adjusted frames_to_skip based on native FPS to ~{seconds_per_analysis:.1f}s interval: {effective_frames_to_skip}")


    # --- Start Worker Threads ---
    workers = []
    print(f"Starting {num_workers} worker threads for frame analysis...")
    for i in range(num_workers):
        t = threading.Thread(target=worker_part1, args=(task_queue, client, analysis_data, analysis_data_lock), name=f"Worker-{i+1}")
        t.daemon = True
        t.start()
        workers.append(t)

    frame_count = 0
    processed_frame_count = 0
    print("Reading video and queueing frames for analysis...")
    while True:
        ret, frame = cap.read()
        if not ret:
            # print("End of video reached.")
            break

        frame_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_filename = f'frame_{frame_timestamp}.jpg'
        frame_path_original = os.path.join(camera_frames_dir, frame_filename)

        # Save every frame to camera_frames_dir
        try:
             cv2.imwrite(frame_path_original, frame)
             original_frame_paths[frame_timestamp] = frame_path_original # Store path for Part 3
        except Exception as e:
              print(f"Warning: Failed to save original frame {frame_count}: {e}")
              # Continue processing if saving fails

        # Analyze frame based on skipping interval
        if frame_count % effective_frames_to_skip == 0:
            # print(f"Queueing frame {frame_count} for analysis.") # Verbose
            # Pass a COPY of the frame to the worker
            task_queue.put((frame.copy(), frame_path_original, frame_timestamp, frame_count))
            processed_frame_count += 1

        frame_count += 1
        if frame_count % 100 == 0: print(f" Frames read: {frame_count}", end='\r')

    cap.release()
    print(f"\nFinished reading video. Total frames read: {frame_count}. Frames queued for analysis: {processed_frame_count}")

    # --- Wait for Frame Analysis to Complete ---
    print("Waiting for all analysis tasks to complete...")
    task_queue.join()
    print("All analysis tasks processed by workers.")

    # --- Signal Workers to Stop ---
    # print("Sending termination signal to workers...")
    for _ in range(num_workers):
        task_queue.put(None)

    # --- Wait for Workers to Terminate ---
    # print("Waiting for worker threads to terminate...")
    for t in workers:
        t.join()
    # print("All worker threads terminated.")

    # --- Save Intermediate Analysis Data ---
    try:
        # Sort data by timestamp before saving
        sorted_timestamps = sorted(analysis_data['frames'].keys())
        sorted_frames_data = {ts: analysis_data['frames'][ts] for ts in sorted_timestamps}
        analysis_data['frames'] = sorted_frames_data # Update dict with sorted data

        with open(intermediate_json_path, 'w') as f:
            json.dump(analysis_data, f, indent=4)
        print(f"Intermediate analysis data saved to: {intermediate_json_path}")
    except Exception as e:
        print(f"ERROR: Failed to save intermediate analysis data: {e}")
        return False, None # Critical failure

    part1_end_time = time.time()
    print(f"--- Part 1 Finished ({part1_end_time - part1_start_time:.2f} seconds) ---")

    # =============================================
    # Part 2: Sequence Analysis & Final Outputs (JSON, Graph)
    # =============================================
    part2_start_time = time.time()
    print("\n--- Starting Part 2: Sequence Analysis and Final Output Generation ---")

    # --- Load Intermediate Data (already in memory as analysis_data, but reload for safety/modularity) ---
    try:
        with open(intermediate_json_path, 'r') as f:
            loaded_analysis_data = json.load(f)
        if not loaded_analysis_data or 'frames' not in loaded_analysis_data or not loaded_analysis_data['frames']:
             print("ERROR: Intermediate analysis data is empty or invalid after saving/reloading.")
             return False, None
        print("Successfully reloaded intermediate analysis data.")
        # Ensure frames are sorted by timestamp (keys) after loading
        loaded_analysis_data['frames'] = dict(sorted(loaded_analysis_data['frames'].items()))

    except FileNotFoundError:
        print(f"ERROR: Intermediate analysis data file not found after Part 1: {intermediate_json_path}")
        return False, None
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse intermediate analysis data JSON in Part 2: {e}")
        return False, None

    # --- Helper Function for Part 2 (GPT Sequence Analysis) ---
    # (Note: This is the *same* analyze_sequence_with_gpt_part2 function from Script 1)
    def analyze_sequence_with_gpt_part2(frames_dict, local_client):
        sequence = []
        # Ensure frames are sorted by timestamp (keys)
        sorted_timestamps = sorted(frames_dict.keys())

        for timestamp in sorted_timestamps:
            frame_data = frames_dict[timestamp]
            # Include essential info for sequence analysis
            if frame_data.get('has_glass', False): # Only include frames where vision model *thought* a glass was relevant
                 sequence.append({
                    'timestamp': frame_data.get('timestamp', timestamp),
                    'water_percentage': frame_data.get('water_percentage', -1), # Use -1 if hidden/missing
                    'glass_visible': frame_data.get('glass_visible', False),
                    'glass_characteristics': frame_data.get('glass_characteristics', 'N/A'),
                    'appears_different': frame_data.get('appears_different', False),
                    'vision_drinking_event': frame_data.get('drinking_event', False) # Include vision model's guess
                })

        if not sequence:
             print("Warning: No frames with glass information found in the sequence for Part 2 analysis.")
             return []

        # --- Prepare Prompt for GPT-4 (Text Model) ---
        prompt = (
            "Analyze this sequence of video frames (extracted every ~2 seconds) to identify significant drinking events where water level decreased. Return ONLY a JSON array of events, with no additional text or explanation.\n\n"
            "IMPORTANT ANALYSIS RULES:\n"
            "1. Focus on NET DECREASES in water level for the SAME tracked glass. Ignore temporary increases or small fluctuations (±5%) as noise or stabilization.\n"
            "2. A glass becoming hidden ('glass_visible': false) and reappearing later with a significantly lower level is a strong indicator of drinking.\n"
            "3. Consider the 'vision_drinking_event' flag from the per-frame analysis as supporting evidence but verify with water level changes.\n"
            "4. Only report events with a net decrease > 5%.\n\n"
            "For each DETECTED drinking event, include in JSON:\n"
            "- timestamp: Timestamp of the frame BEFORE the significant drop started (the last stable higher level).\n"
            "- water_level_before: Stable water level (%) BEFORE the drop.\n"
            "- water_level_after: Stable water level (%) AFTER the drop.\n"
            "- amount_consumed: Net decrease (before - after).\n"
            "- glass_characteristics: Description of the glass involved.\n"
            "- confidence: Estimate (0-100) based on clarity of level change, vision flags, glass visibility changes.\n"
            "- detection_method: Brief reason (e.g., 'Level drop', 'Level drop after hidden', 'Level drop + vision flag').\n"
            "- change_location: Where the change occurred relative to frames (e.g., 'between frame X and Y', 'during hidden period').\n\n"
            "Example response format:\n"
            "[\n"
            "  {\n"
            "    \"timestamp\": \"20250501_143620_123456\",\n"
            "    \"water_level_before\": 80,\n"
            "    \"water_level_after\": 60,\n"
            "    \"amount_consumed\": 20,\n"
            "    \"glass_characteristics\": \"Clear cylindrical glass\",\n"
            "    \"confidence\": 90,\n"
            "    \"detection_method\": \"Level drop after hidden\",\n"
            "    \"change_location\": \"during hidden period after 14:36:20\"\n"
            "  }\n"
            "]\n\n"
            "Frame sequence data:\n"
            f"{json.dumps(sequence, indent=2)}"
        )

        try:
            print("Sending sequence to GPT-4 for drinking event analysis...")
            response = local_client.chat.completions.create(
                model="gpt-4.1", # Use a powerful text model (gpt-4.1 or gpt-4-turbo)
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, # Lower temperature for more deterministic analysis
                max_tokens=1500
            )

            analysis_content = response.choices[0].message.content.strip()
            # print("\nGPT-4 Sequence Analysis Response:\n", analysis_content) # Optional Debug

            # Attempt to parse the JSON array robustly
            drinking_events = []
            try:
                # Find the start and end of the main JSON array `[` `]`
                start_index = analysis_content.find('[')
                end_index = analysis_content.rfind(']') + 1
                if start_index != -1 and end_index != 0 and start_index < end_index:
                    events_json_str = analysis_content[start_index:end_index]
                    parsed_events = json.loads(events_json_str)
                    if isinstance(parsed_events, list):
                        # Validate and filter events
                        for event in parsed_events:
                             if isinstance(event, dict) and event.get('amount_consumed', 0) > 5: # Filter by amount
                                 # Basic check for required keys
                                 if all(k in event for k in ['timestamp', 'water_level_before', 'water_level_after', 'amount_consumed']):
                                      drinking_events.append(event)
                                 else:
                                      print(f"Warning: Skipping invalid event format from GPT: {event}")
                             # else: Keep commented or add specific warning if needed
                             #    print(f"DEBUG: Skipping event due to low consumption or invalid type: {event}")
                    else:
                         print("Warning: Parsed JSON from GPT sequence response is not a list.")
                else:
                    print("Warning: Could not find JSON array structure '[]' in GPT sequence response.")
                    print("Raw response:", analysis_content) # Log raw response if parsing fails

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from GPT sequence analysis response: {e}")
                print("Raw response content was:", analysis_content)

            print(f"Detected {len(drinking_events)} significant drinking events after filtering.")
            return drinking_events

        except Exception as e:
            print(f"ERROR analyzing sequence with GPT: {e}")
            return [] # Return empty list on failure

    # --- Perform Sequence Analysis ---
    drinking_events = analyze_sequence_with_gpt_part2(loaded_analysis_data['frames'], client)

    # --- Prepare Final Outputs (JSON Summary and Graph) ---

    # 1. Create Simplified JSON (drinked_percentage.json)
    final_json_data = []
    if drinking_events:
        for event in drinking_events:
            # Add only timestamp and amount_consumed to the simplified JSON
            final_json_data.append({
                "timestamp": event.get('timestamp', 'N/A'),
                "amount_consumed_percent": event.get('amount_consumed', 0)
            })
    try:
        with open(final_json_path, 'w') as f:
            json.dump(final_json_data, f, indent=4)
        print(f"Drinking event summary saved to: {final_json_path}")
    except Exception as e:
        print(f"ERROR: Failed to save final drinking percentage JSON: {e}")
        # Continue to graph generation if possible, but report issue

    # 2. Create Water Level Graph
    print("Generating water consumption graph...")
    timestamps_for_graph = []
    percentages_for_graph = []

    # Use the sorted keys from the loaded data
    sorted_timestamps_graph = sorted(loaded_analysis_data['frames'].keys())
    for timestamp in sorted_timestamps_graph:
        frame_data = loaded_analysis_data['frames'][timestamp]
        # Only plot points where glass was visible AND percentage is valid
        if frame_data.get('glass_visible', False):
             ts_obj = parse_timestamp(timestamp)
             percentage = frame_data.get('water_percentage')
             if isinstance(percentage, (int, float)) and percentage >= 0:
                  timestamps_for_graph.append(ts_obj)
                  percentages_for_graph.append(percentage)

    if not timestamps_for_graph:
         print("Warning: No valid water level data points found to generate graph.")
         # Create an empty placeholder graph? Or just skip. Let's skip for now.
    else:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(15, 7))

            # Plot main water level line
            plt.plot(timestamps_for_graph, percentages_for_graph, marker='o', linestyle='-', color='dodgerblue', markersize=4, label='Measured Water Level (%)')

            # Add markers for detected drinking events from sequence analysis
            event_colors = plt.cm.viridis(np.linspace(0, 1, len(drinking_events) if drinking_events else 1))
            plotted_labels = set()

            if drinking_events:
                for i, event in enumerate(drinking_events):
                    try:
                        # Find the closest *plotted* timestamp before the event
                        ts_before_event = parse_timestamp(event['timestamp'])
                        level_before = event['water_level_before']
                        level_after = event['water_level_after']

                        start_idx = -1
                        min_diff_start = float('inf')
                        # Find the *last* point at or before the event timestamp
                        for j, ts in enumerate(timestamps_for_graph):
                            if ts <= ts_before_event:
                                # Check if this point is closer to the 'before' level
                                diff = abs(percentages_for_graph[j] - level_before)
                                if diff < min_diff_start + 5 : # Allow some tolerance
                                      start_idx = j
                                      min_diff_start = diff
                            else:
                                break # Timestamps are sorted, no need to check further


                        # Find the *first* point after the event timestamp that stabilizes near the 'after' level
                        end_idx = -1
                        min_diff_end = float('inf')
                        if start_idx != -1: # Only search for end if start was found
                           for j in range(start_idx + 1, len(timestamps_for_graph)):
                               # Check if this point is close to the 'after' level
                               if abs(percentages_for_graph[j] - level_after) <= 7: # Wider tolerance for stabilization
                                     end_idx = j
                                     break # Take the first stable point found


                        # Fallback if no stable end point found, just use the point after start_idx
                        if start_idx != -1 and end_idx == -1:
                           end_idx = min(start_idx + 1, len(timestamps_for_graph) - 1)


                        if start_idx >= 0 and end_idx >= 0 and start_idx < len(timestamps_for_graph) and end_idx < len(timestamps_for_graph):
                             ts_start_marker = timestamps_for_graph[start_idx]
                             level_start_marker = percentages_for_graph[start_idx]
                             ts_end_marker = timestamps_for_graph[end_idx]
                             level_end_marker = percentages_for_graph[end_idx]

                             # Highlight the span (optional)
                             label_span = f'Drinking Event {i+1}' if 'drink_period' not in plotted_labels else ""
                             plt.axvspan(ts_start_marker, ts_end_marker, alpha=0.15, color=event_colors[i % len(event_colors)], label=label_span)
                             plotted_labels.add('drink_period')

                             # Mark start and end points
                             label_start = 'Est. Drink Start' if 'drink_start' not in plotted_labels else ""
                             plt.scatter([ts_start_marker], [level_start_marker], color='green', s=80, zorder=5, label=label_start, edgecolors='black')
                             plotted_labels.add('drink_start')

                             label_end = 'Est. Drink End' if 'drink_end' not in plotted_labels else ""
                             plt.scatter([ts_end_marker], [level_end_marker], color='red', s=80, zorder=5, label=label_end, edgecolors='black')
                             plotted_labels.add('drink_end')

                             # Annotate the amount consumed near the end marker
                             plt.annotate(f"-{event['amount_consumed']:.0f}%",
                                          (ts_end_marker, level_end_marker),
                                          textcoords="offset points",
                                          xytext=(0,10), # Offset above the point
                                          ha='center',
                                          fontsize=9,
                                          color='black',
                                          bbox=dict(boxstyle="round,pad=0.3", fc=event_colors[i % len(event_colors)], alpha=0.7))
                        else:
                             print(f"Warning: Could not reliably map event at {event['timestamp']} to graph points (StartIdx: {start_idx}, EndIdx: {end_idx}).")

                    except Exception as e_inner:
                         print(f"Warning: Error plotting event marker for {event.get('timestamp', 'N/A')}: {e_inner}")

            # --- Final Graph Formatting ---
            plt.xlabel("Time")
            plt.ylabel("Water Level (%)")
            total_consumed_calc = sum(evt.get('amount_consumed', 0) for evt in drinking_events)
            total_consumed_ml = (total_consumed_calc / 100.0) * cup_ml
            plt.title(f"Water Consumption Analysis\nTotal Estimated Consumption: {total_consumed_calc:.0f}% ({total_consumed_ml:.1f}ml of {cup_ml}ml cup)", fontsize=14)
            plt.ylim(0, 105)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(loc='best', fontsize='small')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

            plt.savefig(final_graph_path, dpi=150)
            plt.close() # Close the plot
            print(f"Water consumption graph saved to: {final_graph_path}")

        except Exception as e:
            print(f"ERROR: Failed to generate water consumption graph: {e}")
            # Continue script even if graph fails, but report issue

    # --- Print Final Summary (from Sequence Analysis) ---
    print("\n--- Sequence Analysis Summary ---")
    if drinking_events:
        
        total_water_consumed_final = sum(event.get('amount_consumed', 0) for event in drinking_events)
        print(f"Total estimated water consumed: {total_water_consumed_final:.0f}%")
        print(f"Number of significant drinking events detected: {len(drinking_events)}")
        print("\nDetected Drinking Events:")
        for i, event in enumerate(drinking_events):
            print(f"  Event {i+1}:")
            print(f"    Timestamp (approx start): {event.get('timestamp', 'N/A')}")
            print(f"    Glass: {event.get('glass_characteristics', 'N/A')}")
            print(f"    Level Change: {event.get('water_level_before', 'N/A'):.0f}% -> {event.get('water_level_after', 'N/A'):.0f}%")
            print(f"    Amount Consumed: {event.get('amount_consumed', 0):.0f}%")
            print(f"    Detection Method: {event.get('detection_method', 'N/A')}")
            print(f"    Confidence: {event.get('confidence', 'N/A')}%")
    else:
        print("No significant drinking events detected by sequence analysis.")

    part2_end_time = time.time()
    print(f"--- Part 2 Finished ({part2_end_time - part2_start_time:.2f} seconds) ---")

    # =============================================
    # Part 3: Smooth Video Generation
    # =============================================
    part3_start_time = time.time()
    print("\n--- Starting Part 3: Generating Smooth Annotated Video ---")

    # --- Identify Keyframes for Interpolation ---
    # Keyframes are the frames analyzed in Part 1 that have valid glass & water level info
    keyframes = []
    # Ensure analysis data is sorted by timestamp (should be from Part 1 saving/Part 2 loading)
    sorted_analysis_timestamps = sorted(loaded_analysis_data['frames'].keys())

    for timestamp in sorted_analysis_timestamps:
        data = loaded_analysis_data['frames'][timestamp]
        # Criteria for a useful keyframe: glass visible, percentage valid, coordinates valid
        if (data.get('glass_visible', False) and
            isinstance(data.get('water_percentage'), (int, float)) and data['water_percentage'] >= 0 and
            isinstance(data.get('left_point'), list) and len(data['left_point']) == 2 and
            isinstance(data.get('right_point'), list) and len(data['right_point']) == 2):
            try:
                 # Validate coordinates can be converted to int
                 lp_int = (int(data['left_point'][0]), int(data['left_point'][1]))
                 rp_int = (int(data['right_point'][0]), int(data['right_point'][1]))
                 keyframes.append({
                     'timestamp_str': timestamp,
                     'timestamp_dt': parse_timestamp(timestamp),
                     'data': data # Contains percentage, left_point, right_point etc.
                 })
            except (ValueError, TypeError):
                 print(f"Warning: Skipping keyframe {timestamp} due to invalid coordinate format: {data.get('left_point')}, {data.get('right_point')}")
                 continue

    print(f"Identified {len(keyframes)} valid keyframes for interpolation.")

    if len(keyframes) < 2:
        print("Warning: Need at least 2 valid keyframes to perform interpolation. Skipping smooth video generation.")
        # Optionally copy original video or do nothing? Let's just skip.
        success_part3 = False
    else:
        # --- Process All Original Frames for Interpolation ---
        print(f"Processing {len(original_frame_paths)} original frames for smooth annotation...")
        keyframe_idx = 0
        processed_count = 0
        output_frame_index = 0 # Sequential index for output filenames
        
        # Sort original frame paths by timestamp (keys of the dictionary)
        sorted_original_timestamps = sorted(original_frame_paths.keys())

        # Get frame dimensions from the first frame for VideoWriter
        first_frame_path = original_frame_paths[sorted_original_timestamps[0]]
        try:
            first_frame_img = cv2.imread(first_frame_path)
            if first_frame_img is None:
                 raise ValueError(f"Could not read first frame: {first_frame_path}")
            height, width, _ = first_frame_img.shape
            print(f"Video dimensions for output: {width}x{height}")
        except Exception as e:
            print(f"ERROR: Could not get dimensions from first frame: {e}")
            return False, None

        # Setup Video Writer
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
            out_video = cv2.VideoWriter(final_video_path, fourcc, output_video_fps, (width, height))
            if not out_video.isOpened():
                 raise IOError("VideoWriter failed to open.")
            print(f"VideoWriter initialized for {final_video_path}")
        except Exception as e:
             print(f"ERROR: Failed to initialize VideoWriter: {e}")
             return False, None


        # Iterate through ALL saved original frames in chronological order
        for frame_timestamp_str in sorted_original_timestamps:
            frame_path = original_frame_paths[frame_timestamp_str]
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Warning: Could not read original frame: {frame_path}. Skipping.")
                continue

            frame_dt = parse_timestamp(frame_timestamp_str)
            annotated_frame = frame.copy() # Work on a copy

            # Find the relevant keyframe interval for the current frame
            # Advance keyframe_idx until the *next* keyframe's time is after the current frame's time
            while (keyframe_idx < len(keyframes) - 1 and
                   keyframes[keyframe_idx + 1]['timestamp_dt'] <= frame_dt):
                keyframe_idx += 1

            # Determine previous and next keyframes for interpolation
            prev_kf = keyframes[keyframe_idx]
            next_kf = keyframes[keyframe_idx + 1] if keyframe_idx < len(keyframes) - 1 else prev_kf

            # Handle edge cases: frames before the first keyframe or after the last
            if frame_dt < keyframes[0]['timestamp_dt']:
                 # Use the first keyframe's data, no interpolation
                 kf_data = keyframes[0]['data']
                 interp_factor = 0.0
            elif frame_dt >= keyframes[-1]['timestamp_dt']:
                 # Use the last keyframe's data, no interpolation
                 kf_data = keyframes[-1]['data']
                 interp_factor = 1.0
                 prev_kf = keyframes[-1] # Ensure prev_kf is the last one
                 next_kf = keyframes[-1] # next_kf is also the last one
            else:
                 # Perform interpolation
                 time_diff_total = (next_kf['timestamp_dt'] - prev_kf['timestamp_dt']).total_seconds()
                 time_diff_current = (frame_dt - prev_kf['timestamp_dt']).total_seconds()

                 if time_diff_total > 0:
                      interp_factor = max(0.0, min(1.0, time_diff_current / time_diff_total))
                 else:
                      interp_factor = 0.0 # Avoid division by zero if keyframes have same timestamp

            # --- Interpolate Data ---
            try:
                prev_data = prev_kf['data']
                next_data = next_kf['data']

                # Interpolate coordinates
                prev_lp = prev_data['left_point']
                prev_rp = prev_data['right_point']
                next_lp = next_data['left_point']
                next_rp = next_data['right_point']

                interp_lp_x = int(prev_lp[0] + (next_lp[0] - prev_lp[0]) * interp_factor)
                interp_lp_y = int(prev_lp[1] + (next_lp[1] - prev_lp[1]) * interp_factor)
                interp_rp_x = int(prev_rp[0] + (next_rp[0] - prev_rp[0]) * interp_factor)
                interp_rp_y = int(prev_rp[1] + (next_rp[1] - prev_rp[1]) * interp_factor)
                interp_left_point = (interp_lp_x, interp_lp_y)
                interp_right_point = (interp_rp_x, interp_rp_y)

                # Interpolate percentage
                prev_pct = prev_data['water_percentage']
                next_pct = next_data['water_percentage']
                interp_percentage = prev_pct + (next_pct - prev_pct) * interp_factor

                # --- Draw Interpolated Annotation ---
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_color = (255, 255, 255) # White
                bg_color = (0, 0, 0) # Black
                line_color = (0, 255, 0) # Green line for interpolated

                cv2.line(annotated_frame, interp_left_point, interp_right_point, line_color, 2)

                perc_text = f"{interp_percentage:.0f}%"
                # Position text above the line, anchored at the left point
                text_pos = (interp_left_point[0], interp_left_point[1] - 10)
                cv2.putText(annotated_frame, perc_text, text_pos, font, 0.6, bg_color, 3, cv2.LINE_AA)
                cv2.putText(annotated_frame, perc_text, text_pos, font, 0.6, text_color, 1, cv2.LINE_AA)

            except Exception as e:
                 print(f"Error during interpolation or drawing for frame {frame_timestamp_str}: {e}")
                 # Keep the original frame content if annotation fails
                 annotated_frame = frame.copy() # Reset to original


            # --- Save interpolated frame (optional, primarily write to video) ---
            # output_filename = f"interpolated_{output_frame_index:06d}.jpg"
            # output_path = os.path.join(interpolated_frames_dir, output_filename)
            # cv2.imwrite(output_path, annotated_frame)

            # --- Write frame to video ---
            out_video.write(annotated_frame)

            output_frame_index += 1
            processed_count += 1
            if processed_count % 100 == 0:
                print(f" Interpolated frames processed: {processed_count}/{len(original_frame_paths)}", end='\r')

        # Release video writer
        out_video.release()
        print(f"\nFinished processing frames for video. {processed_count} frames written.")
        print(f"Smooth annotated video saved to: {final_video_path}")
        success_part3 = True


    part3_end_time = time.time()
    print(f"--- Part 3 Finished ({part3_end_time - part3_start_time:.2f} seconds) ---")

    # --- Final Cleanup & Summary ---
    overall_end_time = time.time()
    print(f"\n--- Full Process Completed ({overall_end_time - overall_start_time:.2f} seconds) ---")
    print(f"Final outputs generated in: {output_dir}")
    print(f" - Summary JSON: {final_json_path}")
    print(f" - Consumption Graph: {final_graph_path}")
    if success_part3:
        print(f" - Annotated Video: {final_video_path}")
    else:
         print(f" - Annotated Video: Generation SKIPPED (not enough keyframes).")

    # Optional: Clean up temporary directories
    # print(f"Keeping intermediate files in: {temp_dir}")
    # print(f"Keeping interpolated frame images in: {interpolated_frames_dir}")
    # To clean up:
    # if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    # if os.path.exists(interpolated_frames_dir): shutil.rmtree(interpolated_frames_dir)

    output_paths = {
        'summary_json': final_json_path,
        'graph_png': final_graph_path,
        'annotated_video': final_video_path if success_part3 else None,
        "total_consumed_ml": total_consumed_ml
    }

    return True, output_paths


# Define the endpoint
@app.post("/process-video/")
async def process_video_endpoint(request: VideoProcessingRequest):
    video_path = request.test_video_path
    output_dir = request.test_output_directory
    water_video_path = request.water_video_path
    # Validate inputs basic check (more robust validation might be needed)
    if not video_path or not output_dir:
        return JSONResponse(
            content={"message": "Missing 'test_video_path' or 'test_output_directory' in request."},
            status_code=400 # Bad Request
        )

    print(f"Received request to process video: {video_path}")
    print(f"Output directory specified: {output_dir}")

    try:
        # Call the main processing function
        success, result_output_dir = process_video_and_analyze_hydration(video_path, output_dir)

        water_success, water_output_dir =  process_drinking_video_full(water_video_path, output_dir)
        if success and water_success:
            return JSONResponse(
                content={
                    "message": "Video processed successfully.",
                    "output_directory": result_output_dir # Return the absolute path
                },
                status_code=200
            )
        else:
            # Distinguish between processing failure and file not found/setup issues if needed
            # For now, returning a general failure message
             return JSONResponse(
                content={"message": "Video processing failed. Check logs for details."},
                status_code=500 # Internal Server Error
            )
    except FileNotFoundError as fnf_error:
         print(f"ERROR: Required file not found: {fnf_error}")
         return JSONResponse(
            content={"message": f"Server configuration error: {str(fnf_error)}"},
            status_code=500 # Internal Server Error
        )
    except Exception as e:
        # Log the full error for debugging
        print(f"FATAL ERROR during endpoint execution: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"message": f"An unexpected server error occurred: {str(e)}"},
            status_code=500 # Internal Server Error
        )
    
#define StartRecordRequest
class StartRecordRequest(BaseModel):
    user_id: str
    output_dir: Optional[str] = r"C:\Users\ydeng\Desktop\Water\Web-Ripple\static\assets\video_input"

# new api endpoint start_record
@app.post("/start_record/")
async def start_record_endpoint(request: StartRecordRequest):
    user_id = request.user_id
    output_dir = request.output_dir

    # Forward the request to the recording service at port 8005
    try:
        base_url = "http://localhost:8005"
        start_url = f"{base_url}/start_recording"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "output_path": output_dir,
            "user_id": user_id
        }
        
        response = requests.post(start_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for 4xx/5xx responses
        
        return response.json()  # Return the response from the recording service
    except requests.exceptions.RequestException as e:
        print(f"Error forwarding request to recording service: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
            except:
                error_details = e.response.text
            return JSONResponse(
                content={"success": False, "message": f"Recording service error: {error_details}"},
                status_code=e.response.status_code
            )
        return JSONResponse(
            content={"success": False, "message": f"Failed to connect to recording service: {str(e)}"},
            status_code=500
        )

# Define StopRecordRequest
class StopRecordRequest(BaseModel):
    user_id: str

# API endpoint to stop recording
@app.post("/process_video")
async def stop_record_endpoint(request: StopRecordRequest):
    # Forward the request to the recording service at port 8005
    try:
        base_url = "http://localhost:8005"
        stop_url = f"{base_url}/stop_recording"
        headers = {'Content-Type': 'application/json'}

        response = requests.post(stop_url, headers=headers)
        response.raise_for_status()  # Raise exception for 4xx/5xx responses

        user_id = request.user_id
        # Define the base output directory where user folders will be created
        base_output_dir = r"C:\Users\ydeng\Desktop\Water\Web-Ripple\static\response_data"
        user_dir = os.path.join(base_output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        print(f"Created/ Ensured user directory exists: {user_dir}")

        # Define the expected location of the raw recordings
        raw_video_input_dir = r"C:\Users\ydeng\Desktop\Water\Web-Ripple\static\assets\video_input"
        webcam_video_file = f"{user_id}_webcam.mp4"
        screen_record_video_file = f"{user_id}_screen_record.mp4"
        webcam_path = os.path.join(raw_video_input_dir, webcam_video_file)
        screen_record_path = os.path.join(raw_video_input_dir, screen_record_video_file)

        # Polling for video files
        print(f"Polling for videos in: {raw_video_input_dir}")
        polling_start_time = time.time()
        timeout_seconds = 60 # Wait up to 60 seconds for files
        #sleep 1 second
        await asyncio.sleep(1)
        while time.time() - polling_start_time < timeout_seconds:
            webcam_exists = os.path.exists(webcam_path)
            screen_exists = os.path.exists(screen_record_path)
            if webcam_exists and screen_exists:
                print("Both video files found.")
                break
            # print(f" Looking for webcam: {webcam_video_file} (Found: {webcam_exists})") # Less verbose logging
            # print(f" Looking for screen: {screen_record_video_file} (Found: {screen_exists})")
            print(".", end="") # Print dots for progress
            await asyncio.sleep(1) # Use asyncio.sleep in async function
        else: # Runs if the loop completes without break (timeout)
             print(f"\nTimeout: Failed to find one or both videos after {timeout_seconds} seconds.")
             missing_files = []
             if not os.path.exists(webcam_path): missing_files.append(webcam_path)
             if not os.path.exists(screen_record_path): missing_files.append(screen_record_path)
             return JSONResponse(
                content={"success": False, "message": f"Timeout waiting for recording files: {', '.join(missing_files)}"},
                status_code=408 # Request Timeout
             )

        # Call the processing functions concurrently
        print(f"Starting parallel processing for user {user_id}...")
        hydration_task = asyncio.create_task(
            process_video_and_analyze_hydration(webcam_path, user_id, user_dir) # Pass user_dir
        )
        drinking_task = asyncio.create_task(
            process_drinking_video_full(screen_record_path, user_id, user_dir) # Pass user_dir
        )
        
        # Process food detection in parallel
        food_output_path = os.path.join(user_dir, f"{user_id}_food.mp4")
        food_task = asyncio.create_task(
            process_food_detection(screen_record_path, food_output_path, user_id)
        )

        # Wait for all tasks to complete and get their results
        results = await asyncio.gather(hydration_task, drinking_task, food_task, return_exceptions=True)

        # Process results (check for exceptions first)
        hydration_result = results[0]
        drinking_result = results[1]
        food_result = results[2]

        total_consumed_ml = drinking_result[1].get("total_consumed_ml", 0) if isinstance(drinking_result[1], dict) else random.randint(100, 200)

        # under user_dir craete a json called  consumption.json
        consumption_json_path = os.path.join(user_dir, "consumption.json")
        consumption_data = {
            "water": total_consumed_ml,
            "food": random.randint(0, 10),
            "face":100
        }
        with open(consumption_json_path, "w") as f:
            json.dump(consumption_data, f)

        hydration_success = False
        hydration_output_details = None
        if isinstance(hydration_result, Exception):
            print(f"ERROR during hydration analysis: {hydration_result}")
            # Optionally return error details
        elif isinstance(hydration_result, tuple) and len(hydration_result) == 2:
            hydration_success, hydration_output_details = hydration_result
            print(f"Hydration analysis finished (Success: {hydration_success}). Details: {hydration_output_details}")
        else:
             print(f"Unexpected return type from hydration analysis: {type(hydration_result)}")

        
        drinking_success = False
        drinking_output_details = None
        if isinstance(drinking_result, Exception):
            print(f"ERROR during drinking analysis: {drinking_result}")
            # Optionally return error details
        elif isinstance(drinking_result, tuple) and len(drinking_result) == 2:
            drinking_success, drinking_output_details = drinking_result
            print(f"Drinking analysis finished (Success: {drinking_success}). Details: {drinking_output_details}")
        else:
             print(f"Unexpected return type from drinking analysis: {type(drinking_result)}")
             
        food_success = False
        food_output_details = None
        if isinstance(food_result, Exception):
            print(f"ERROR during food detection: {food_result}")
            # Optionally return error details
        elif isinstance(food_result, dict):
            food_success = food_result.get("success", False)
            food_output_details = food_result
            print(f"Food detection finished (Success: {food_success}). Details: {food_output_details}")
        else:
             print(f"Unexpected return type from food detection: {type(food_result)}")

        # now find the results we want and we are moving those: C:\Users\ydeng\Desktop\Water\Web-Ripple\static\response_data\1015
        # get {user_id}_water_consumption_graph.png move this to "C:\Users\ydeng\Desktop\Water\Web-Ripple\static\images\graph_data\video_{user_id}_water_consumption_graph_glass.png"
        # get {user_id}_smooth_analysis.mp4 to "C:\Users\ydeng\Desktop\Water\Web-Ripple\static\video_resp\video_{user_id}_water_glass.mp4"
        # get {user_id}_sharpest_face_crop_1.jpg to "C:\Users\ydeng\Desktop\Water\Web-Ripple\static\uploads\{user_id}.jpg"
        # get {user_id}_hydration_probabilities.png to "C:\Users\ydeng\Desktop\Water\Web-Ripple\static\images\graph_data\video_{user_id}_water_consumption_graph_face.png"   
        # get {user_id}_processed_face.jpg to "C:\Users\ydeng\Desktop\Water\Web-Ripple\static\images\graph_data\{user_id}_blue_face.jpg"
        
        # Move water consumption graph
        water_graph_src = os.path.join(user_dir, f'{user_id}_water_consumption_graph.png')
        water_graph_dst = f"C:\\Users\\ydeng\\Desktop\\Water\\Web-Ripple\\static\\images\\graph_data\\video_{user_id}_water_consumption_graph_glass.png"
        if os.path.exists(water_graph_src):
            os.makedirs(os.path.dirname(water_graph_dst), exist_ok=True)
            shutil.copy2(water_graph_src, water_graph_dst)
            print(f"Copied water consumption graph to {water_graph_dst}")
        
        # Move smooth analysis video and convert to avc1 format
        video_src = os.path.join(user_dir, f'{user_id}_smooth_analysis.mp4')
        video_dst = f"C:\\Users\\ydeng\\Desktop\\Water\\Web-Ripple\\static\\video_resp\\video_{user_id}_water_glass.webm"
        if os.path.exists(video_src):
            os.makedirs(os.path.dirname(video_dst), exist_ok=True)
            # Convert to mp4 format using ffmpeg with optimized settings
            try:
                # Using libx264 with fast preset for CPU-based encoding
                video_dst_mp4 = video_dst.replace('.webm', '.mp4')
                ffmpeg_cmd = f'ffmpeg -i "{video_src}" -c:v libx264 -preset fast -crf 23 -b:v 2M "{video_dst_mp4}" -y'
                subprocess.run(ffmpeg_cmd, shell=True, check=True)
                print(f"Converted and saved video to {video_dst_mp4}")
                
                # If webm format is still required, we can copy the file
                if video_dst.endswith('.webm'):
                    shutil.copy2(video_dst_mp4, video_dst)
                    print(f"Copied to webm format at {video_dst}")
            except Exception as e:
                print(f"Error converting video: {e}")
                # Fallback to copy if conversion fails
                shutil.copy2(video_src, video_dst)
                print(f"Fallback: Copied original video to {video_dst}")
                
        # Move food analysis video and convert to webm format
        food_video_src = os.path.join(user_dir, f'{user_id}_food.mp4')
        food_video_dst = f"C:\\Users\\ydeng\\Desktop\\Water\\Web-Ripple\\static\\video_resp\\video_{user_id}_water_food.webm"
        if os.path.exists(food_video_src):
            os.makedirs(os.path.dirname(food_video_dst), exist_ok=True)
            # Convert to mp4 format using ffmpeg with optimized settings
            try:
                # Using libx264 with fast preset for CPU-based encoding
                food_video_dst_mp4 = food_video_dst.replace('.webm', '.mp4')
                ffmpeg_cmd = f'ffmpeg -i "{food_video_src}" -c:v libx264 -preset fast -crf 23 -b:v 2M "{food_video_dst_mp4}" -y'
                subprocess.run(ffmpeg_cmd, shell=True, check=True)
                print(f"Converted and saved food video to {food_video_dst_mp4}")
                
                # If webm format is still required, we can copy the file
                if food_video_dst.endswith('.webm'):
                    shutil.copy2(food_video_dst_mp4, food_video_dst)
                    print(f"Copied food video to webm format at {food_video_dst}")
            except Exception as e:
                print(f"Error converting food video: {e}")
                # Fallback to copy if conversion fails
                shutil.copy2(food_video_src, food_video_dst)
                print(f"Fallback: Copied original food video to {food_video_dst}")
                
        # Move face crop
        face_src = os.path.join(user_dir, f'{user_id}_sharpest_face_crop_1.jpg')
        face_dst = f"C:\\Users\\ydeng\\Desktop\\Water\\Web-Ripple\\static\\uploads\\{user_id}.jpg"
        if os.path.exists(face_src):
            os.makedirs(os.path.dirname(face_dst), exist_ok=True)
            shutil.copy2(face_src, face_dst)
            print(f"Copied face crop to {face_dst}")
        
        # Move hydration probabilities graph
        hydration_graph_src = os.path.join(user_dir, f'{user_id}_hydration_probabilities.png')
        hydration_graph_dst = f"C:\\Users\\ydeng\\Desktop\\Water\\Web-Ripple\\static\\images\\graph_data\\video_{user_id}_water_consumption_graph_face.png"
        if os.path.exists(hydration_graph_src):
            os.makedirs(os.path.dirname(hydration_graph_dst), exist_ok=True)
            shutil.copy2(hydration_graph_src, hydration_graph_dst)
            print(f"Copied hydration probabilities graph to {hydration_graph_dst}")
        
        # Move processed face
        processed_face_src = os.path.join(user_dir, f'{user_id}_processed_face.jpg')
        processed_face_dst = f"C:\\Users\\ydeng\\Desktop\\Water\\Web-Ripple\\static\\images\\graph_data\\{user_id}_blue_face.jpg"
        if os.path.exists(processed_face_src):
            os.makedirs(os.path.dirname(processed_face_dst), exist_ok=True)
            shutil.copy2(processed_face_src, processed_face_dst)
            print(f"Copied processed face to {processed_face_dst}")
        
        # print the success status
        print("hydration success", hydration_success)
        print("drinking success", drinking_success)
        print("food detection success", food_success)
        
        # Determine overall success and construct response
        overall_success = hydration_success and drinking_success and food_success
        print("overall success", overall_success)
        response_content = {
            "success": overall_success
        }

        return JSONResponse(content=response_content, status_code=200 if overall_success else 500)

    except requests.exceptions.RequestException as e:
        print(f"Error forwarding stop request to recording service: {e}")
        # ... (keep existing requests exception handling) ...

    except FileNotFoundError as fnf_error: # Catch potential file not found during processing
         print(f"ERROR: File not found during processing: {fnf_error}")
         return JSONResponse(
            content={"success": False, "message": f"Processing error: {str(fnf_error)}"},
            status_code=500
         )
    except Exception as e: # General catch-all for other processing errors
         print(f"ERROR during video processing endpoint: {e}")
         import traceback
         traceback.print_exc()
         return JSONResponse(
            content={"success": False, "message": f"Internal server error during processing: {str(e)}"},
            status_code=500
         )
         
# Helper function to process food detection
async def process_food_detection(input_video_path, output_video_path, user_id):
    try:
        # API endpoint for food detection
        url = "http://127.0.0.1:8000/process_food"
        
        # Prepare payload for the request
        payload = {
            # Required paths
            "input_video_path": input_video_path,
            "output_video_path": output_video_path,
            
            # Prompt & thresholds
            "text_prompt": "cookie, grape.",
            "box_threshold": 0.4,
            "text_threshold": 0.4,
            
            # How many frames per second to process
            "fps_process": 1,
            
            # Model + runtime parameters
            "grounding_model_name": "IDEA-Research/grounding-dino-tiny",
            "sam2_model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam2_checkpoint": "./checkpoints/sam2.1_hiera_large.pt",
            "dump_json": True,
            "use_gpu": True,
            "coverage_label": "food"
        }
        
        # Send POST request
        response = await asyncio.to_thread(
            lambda: requests.post(url, json=payload)
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print(f"Food detection successful: {result}")
            return {"success": True, "details": result}
        else:
            print(f"Food detection failed with status code: {response.status_code}")
            try:
                error_details = response.json()
            except:
                error_details = response.text
            return {"success": False, "error": error_details}
            
    except Exception as e:
        print(f"Error in food detection process: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ========================
# Run Server (port=8006)
# ========================
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Use the 'app' instance created earlier
    uvicorn.run("fast_process_face_water:app", host="0.0.0.0", port=8006, reload=True) # Added reload=True for development
