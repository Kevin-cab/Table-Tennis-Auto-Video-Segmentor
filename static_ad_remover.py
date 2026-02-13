import subprocess
import sys

def auto_install_packages():
    """Automatically install required packages if they're missing."""
    required_packages = {
        'cv2': 'opencv-python',
        'imagehash': 'imagehash',
        'numpy': 'numpy',
        'PIL': 'Pillow'
    }
    
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            print(f"Installing {package_name}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
                print(f"Successfully installed {package_name}")
            except subprocess.CalledProcessError:
                print(f"Failed to install {package_name}. Please install manually: pip install {package_name}")
                sys.exit(1)

# Auto-install required packages
auto_install_packages()

import cv2
import imagehash
import numpy as np
from PIL import Image

# Configuration constants for pHash static ad detection
SAMPLE_EVERY_N_FRAMES = 2
PHASH_HAMMING_THRESHOLD = 1  # Must be nearly identical (0-1 only)
STATIC_RATIO_THRESHOLD = 0.95  # Require 95% identical frames
PIXEL_DIFF_THRESHOLD = 0.01  # Max 1% pixel change for truly static frames

def is_static_ad(video_path, sample_every_n=None):
    """
    Detect if a video clip is a static advertisement using perceptual hashing (pHash).
    
    Args:
        video_path: Path to video file
        sample_every_n: Frame sampling rate (defaults to SAMPLE_EVERY_N_FRAMES)
    
    Returns:
        (is_ad: bool, static_ratio: float, reason: str)
    """
    if sample_every_n is None:
        sample_every_n = SAMPLE_EVERY_N_FRAMES
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 5:
        cap.release()
        return False, 0.0, "Too few frames"

    prev_hash = None
    prev_frame = None
    identical_pairs = 0
    analyzed_pairs = 0
    frame_idx = 0
    hamming_distances = []
    pixel_diffs = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample every N frames
        if frame_idx % sample_every_n != 0:
            frame_idx += 1
            continue

        try:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Downscale for faster processing
            frame_small = cv2.resize(frame_rgb, (320, 240))
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_small)
            
            # Compute perceptual hash
            current_hash = imagehash.phash(pil_image)
            
            if prev_hash is not None and prev_frame is not None:
                # Calculate Hamming distance
                hamming_dist = current_hash - prev_hash
                hamming_distances.append(hamming_dist)
                
                # Calculate pixel difference (normalized)
                diff = cv2.absdiff(frame_small, prev_frame)
                pixel_diff = np.mean(diff) / 255.0  # Normalize to 0-1
                pixel_diffs.append(pixel_diff)
                
                # Count as identical ONLY if both phash AND pixel diff are low
                if hamming_dist <= PHASH_HAMMING_THRESHOLD and pixel_diff <= PIXEL_DIFF_THRESHOLD:
                    identical_pairs += 1
                
                analyzed_pairs += 1
            
            prev_hash = current_hash
            prev_frame = frame_small.copy()
            
        except Exception as e:
            # Skip frame on error
            print(f"Warning: Error processing frame {frame_idx}: {e}")
            pass
        
        frame_idx += 1

    cap.release()

    if analyzed_pairs == 0:
        return False, 0.0, "No pairs analyzed"

    static_ratio = identical_pairs / analyzed_pairs
    is_ad = static_ratio >= STATIC_RATIO_THRESHOLD
    
    # Calculate averages for logging
    avg_hamming = sum(hamming_distances) / len(hamming_distances) if hamming_distances else 0
    avg_pixel_diff = sum(pixel_diffs) / len(pixel_diffs) if pixel_diffs else 0

    reason = f"pHash+pixel identical ratio: {static_ratio:.2%} ({identical_pairs}/{analyzed_pairs}), avg_hamming: {avg_hamming:.1f}, avg_pixel_diff: {avg_pixel_diff:.3f}"
    return is_ad, static_ratio, reason
