import subprocess
import sys

def auto_install_packages():
    """Automatically install required packages if they're missing."""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'torchvision': 'torchvision',
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
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Configuration constants for CNN+flow slow-motion detection
SAMPLE_EVERY_N_FRAMES = 2
SLOWMO_COS_LOW = 0.80  # Mapped from ssim_min
SLOWMO_COS_HIGH = 0.96  # Mapped from ssim_max
SLOWMO_RATIO_THRESHOLD = 0.60  # For detection/diagnostics
SLOWMO_SHOT_REMOVE_THRESHOLD = 0.45  # Mapped from ratio_threshold
FLOW_MEAN_MIN = 0.005  # Mapped from diff_min
FLOW_MEAN_MAX = 0.10  # Mapped from diff_max
FLOW_LOW_FRACTION_THRESHOLD = 0.6

# SAMPLE_EVERY_N_FRAMES = 2
# SLOWMO_COS_LOW = 0.97  # Lowered from 0.98 to catch more variation
# SLOWMO_COS_HIGH = 0.995
# SLOWMO_RATIO_THRESHOLD = 0.60  # For detection/diagnostics
# SLOWMO_SHOT_REMOVE_THRESHOLD = 0.75  # Lowered from 0.85 to catch faster slow-mo
# FLOW_MEAN_MIN = 0.02
# FLOW_MEAN_MAX = 0.6  # Increased from 0.5 to allow slightly faster motion
# FLOW_LOW_FRACTION_THRESHOLD = 0.6

# Global model cache
_resnet_model = None
_device = None

def _get_resnet_model():
    """Get or initialize the ResNet50 embedding model."""
    global _resnet_model, _device
    if _resnet_model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load pretrained ResNet50 and remove final fc layer
        resnet = models.resnet50(pretrained=True)
        _resnet_model = torch.nn.Sequential(*list(resnet.children())[:-1])
        _resnet_model.to(_device)
        _resnet_model.eval()
    return _resnet_model, _device

def _compute_embeddings_batch(frames):
    """
    Compute ResNet50 embeddings for a batch of frames.
    
    Args:
        frames: List of BGR numpy arrays
    
    Returns:
        L2-normalized embeddings as numpy array (N, embed_dim)
    """
    model, device = _get_resnet_model()
    
    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert BGR to RGB and transform
    tensors = []
    for frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = transform(pil_img)
        tensors.append(tensor)
    
    # Batch processing
    batch = torch.stack(tensors).to(device)
    
    with torch.no_grad():
        embeddings = model(batch)
        embeddings = embeddings.squeeze(-1).squeeze(-1)  # Remove spatial dimensions
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()

def _compute_optical_flow_stats(frame1, frame2):
    """
    Compute optical flow statistics between two frames.
    
    Args:
        frame1, frame2: BGR numpy arrays
    
    Returns:
        Normalized mean magnitude
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Downscale for faster processing
    h, w = gray1.shape
    target_size = (320, 240)
    gray1 = cv2.resize(gray1, target_size)
    gray2 = cv2.resize(gray2, target_size)
    
    # Compute dense optical flow using Farneback
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Calculate magnitude
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = np.mean(mag)
    
    # Normalize by diagonal length
    diagonal = np.sqrt(target_size[0]**2 + target_size[1]**2)
    normalized_mag = mean_mag / diagonal
    
    return normalized_mag

def is_slow_motion(video_path, sample_every_n=None):
    """
    Detect if a video clip contains slow-motion replay using CNN embeddings + optical flow.
    
    Args:
        video_path: Path to video file
        sample_every_n: Frame sampling rate (defaults to SAMPLE_EVERY_N_FRAMES)
    
    Returns:
        (is_slowmo: bool, slowmo_ratio: float, reason: str)
    """
    if sample_every_n is None:
        sample_every_n = SAMPLE_EVERY_N_FRAMES
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 10:
        cap.release()
        return False, 0.0, "Too few frames"

    # Sample frames
    sampled_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            sampled_frames.append(frame)

        frame_idx += 1

    cap.release()

    if len(sampled_frames) < 3:
        return False, 0.0, "Too few sampled frames for reliable slow-mo decision"

    # Step 1: Compute embeddings in batch
    embeddings = _compute_embeddings_batch(sampled_frames)
    
    # Step 2: Compute cosine similarities between adjacent pairs
    cos_similarities = []
    for i in range(len(embeddings) - 1):
        cos_sim = np.dot(embeddings[i], embeddings[i + 1])
        cos_similarities.append(cos_sim)
    
    # Step 3: Count slowmo-like pairs (embedding test)
    slowmo_like_pairs = 0
    for cos_sim in cos_similarities:
        if SLOWMO_COS_LOW <= cos_sim < SLOWMO_COS_HIGH:
            slowmo_like_pairs += 1
    
    analyzed_pairs = len(cos_similarities)
    slowmo_ratio = slowmo_like_pairs / analyzed_pairs if analyzed_pairs > 0 else 0.0
    
    # Step 4: Optical flow confirmation
    low_motion_pairs = 0
    flow_magnitudes = []
    
    for i in range(len(sampled_frames) - 1):
        norm_mag = _compute_optical_flow_stats(sampled_frames[i], sampled_frames[i + 1])
        flow_magnitudes.append(norm_mag)
        
        if FLOW_MEAN_MIN <= norm_mag <= FLOW_MEAN_MAX:
            low_motion_pairs += 1
    
    low_motion_fraction = low_motion_pairs / analyzed_pairs if analyzed_pairs > 0 else 0.0
    flow_confirm = low_motion_fraction >= FLOW_LOW_FRACTION_THRESHOLD
    
    # Step 5: Final decision - only remove if strong majority (>= 85%) of pairs are slow-mo
    # This prevents removing shots with only partial slow-motion that may be needed for merging
    is_slowmo = slowmo_ratio >= SLOWMO_SHOT_REMOVE_THRESHOLD
    
    # Statistics for logging
    cos_min = min(cos_similarities) if cos_similarities else 0.0
    cos_max = max(cos_similarities) if cos_similarities else 0.0
    
    reason = (f"slowmo_ratio={slowmo_ratio:.2%}, slowmo_pairs={slowmo_like_pairs}/{analyzed_pairs}, "
              f"low_motion_frac={low_motion_fraction:.2%}, "
              f"cos_stats(min,max)={cos_min:.3f},{cos_max:.3f}, "
              f"flow_confirm={flow_confirm}, remove_threshold={SLOWMO_SHOT_REMOVE_THRESHOLD:.2f}")
    
    return is_slowmo, slowmo_ratio, reason
