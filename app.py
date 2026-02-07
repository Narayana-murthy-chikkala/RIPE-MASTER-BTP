# Optimized RIPE Gradio Interface qwert
# High-performance keypoint extraction with caching, memory optimization, and batching

import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import gc

import gradio as gr
import numpy as np
import torch
import cv2
import kornia.feature as KF
import kornia.geometry as KG
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup
SEED = 32000
os.environ["PYTHONHASHSEED"] = str(SEED)

import random
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# Model and utility imports
from ripe.utils.utils import cv2_matches_from_kornia, to_cv_kpts

# ============================================================================
# CONFIGURATION
# ============================================================================

# Image size constraints
MIN_SIZE = 512
MAX_SIZE = 768

# Keypoint extraction parameters
KEYPOINT_THRESHOLD = 0.5
TOP_K_KEYPOINTS = 2048

# RANSAC parameters
RANSAC_MIN_MATCHES = 8

# Device management
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# ============================================================================
# HTML DESCRIPTION
# ============================================================================

DESCRIPTION_TEXT = """
<p align='center'>
  <h1 align='center'>🌊🌺 ICCV 2025 🌺🌊</h1>
  <p align='center'>
    <a href='https://scholar.google.com/citations?user=ybMR38kAAAAJ'>Johannes Künzel</a> · 
    <a href='https://scholar.google.com/citations?user=5yTuyGIAAAAJ'>Anna Hilsmann</a> · 
    <a href='https://scholar.google.com/citations?user=BCElyCkAAAAJ'>Peter Eisert</a>
  </p>
  <h2 align='center'>
    <a href='???'>Arxiv</a> | 
    <a href='???'>Project Page</a> | 
    <a href='???'>Code</a>
  </h2>
</p>

<br/>
<div align='center'>

### This demo showcases RIPE: Reinforcement Learning on Unlabeled Image Pairs for Robust Keypoint Extraction.

### RIPE is trained without requiring pose or depth supervision. By leveraging reinforcement learning, it learns from unlabeled image pairs.

The demo extracts keypoints using MNN descriptor matching and RANSAC geometric filtering.

</div>
"""

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Singleton for model management with memory optimization."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self):
        """Get model instance with lazy loading."""
        if self._model is None:
            logger.info("Loading RIPE model...")
            from ripe import vgg_hyper
            
            self._model = vgg_hyper()
            self._model = self._model.to(DEVICE)
            self._model.eval()
            logger.info("Model loaded successfully")
        
        return self._model
    
    def clear(self):
        """Clear model from memory."""
        self._model = None
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()


# Global model manager
model_manager = ModelManager()

# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def get_new_image_size(
    image: Image.Image,
    min_size: int = MIN_SIZE,
    max_size: int = MAX_SIZE,
) -> Tuple[int, int]:
    """Get new image size maintaining aspect ratio."""
    width, height = image.size
    aspect_ratio = width / height
    
    if width > height:
        new_width = max(min_size, min(max_size, width))
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max(min_size, min(max_size, height))
        new_width = int(new_height * aspect_ratio)
    
    return (new_width, new_height)


def preprocess_images(
    image1: Image.Image,
    image2: Image.Image,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, str]:
    """Preprocess images for model inference."""
    log_text = "Preprocessing images...\n"
    log_text += f"Original Image 1 size: {image1.size}\n"
    log_text += f"Original Image 2 size: {image2.size}\n"
    
    # Resize with aspect ratio preservation
    new_size1 = get_new_image_size(image1, MIN_SIZE, MAX_SIZE)
    image1 = image1.resize(new_size1, Image.LANCZOS)
    
    new_size2 = get_new_image_size(image2, MIN_SIZE, MAX_SIZE)
    image2 = image2.resize(new_size2, Image.LANCZOS)
    
    log_text += f"Resized Image 1 size: {image1.size}\n"
    log_text += f"Resized Image 2 size: {image2.size}\n"
    
    # Convert to RGB
    image1 = image1.convert('RGB')
    image2 = image2.convert('RGB')
    
    # Save numpy versions for visualization
    image1_np = np.array(image1)
    image2_np = np.array(image2)
    
    # Convert to tensors
    tensor1 = torch.from_numpy(image1_np).permute(2, 0, 1).float() / 255.0
    tensor2 = torch.from_numpy(image2_np).permute(2, 0, 1).float() / 255.0
    
    # Add batch dimension and move to device
    tensor1 = tensor1.unsqueeze(0).to(DEVICE)
    tensor2 = tensor2.unsqueeze(0).to(DEVICE)
    
    return tensor1, tensor2, image1_np, image2_np, log_text


# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

@torch.no_grad()
def extract_keypoints(
    image1: Image.Image,
    image2: Image.Image,
    inl_th: float,
) -> Tuple[str, Optional[Image.Image], Image.Image, Image.Image]:
    """Main inference function for keypoint extraction."""
    try:
        # Load model
        model = model_manager.get_model()
        
        # Preprocess images
        image1_tensor, image2_tensor, image1_np, image2_np, log_text = preprocess_images(image1, image2)
        
        # Extract keypoints
        logger.info("Extracting keypoints...")
        kpts_1, desc_1, score_1 = model.detectAndCompute(
            image1_tensor,
            threshold=KEYPOINT_THRESHOLD,
            top_k=TOP_K_KEYPOINTS,
        )
        kpts_2, desc_2, score_2 = model.detectAndCompute(
            image2_tensor,
            threshold=KEYPOINT_THRESHOLD,
            top_k=TOP_K_KEYPOINTS,
        )
        
        # Handle batch dimension
        if kpts_1.dim() == 3:
            kpts_1 = kpts_1.squeeze(0)
            desc_1 = desc_1[0].squeeze(0) if isinstance(desc_1, list) else desc_1.squeeze(0)
            score_1 = score_1[0].squeeze(0) if isinstance(score_1, list) else score_1.squeeze(0)
        
        if kpts_2.dim() == 3:
            kpts_2 = kpts_2.squeeze(0)
            desc_2 = desc_2[0].squeeze(0) if isinstance(desc_2, list) else desc_2.squeeze(0)
            score_2 = score_2[0].squeeze(0) if isinstance(score_2, list) else score_2.squeeze(0)
        
        log_text += f"Keypoints in image 1: {kpts_1.shape[0]}\n"
        log_text += f"Keypoints in image 2: {kpts_2.shape[0]}\n"
        
        # Match descriptors
        logger.info("Matching descriptors...")
        matcher = KF.DescriptorMatcher("mnn")
        match_dists, match_idxs = matcher(desc_1, desc_2)

        # 🔥 keep strongest matches only
        k = min(200, match_dists.shape[0])
        order = torch.argsort(match_dists.squeeze())
        order = order[:k]

        match_dists = match_dists[order]
        match_idxs = match_idxs[order]

        # 🔥 spatial sanity filtering
        pts1 = kpts_1[match_idxs[:, 0]]
        pts2 = kpts_2[match_idxs[:, 1]]

        spatial_dist = torch.norm(pts1 - pts2, dim=1)
        keep = spatial_dist < 300.0

        match_dists = match_dists[keep]
        match_idxs = match_idxs[keep]

        cv2_matches = cv2_matches_from_kornia(match_dists, match_idxs)
        
        log_text += f"MNN matches: {match_idxs.shape[0]}\n"
        
        # RANSAC filtering
        do_ransac = match_idxs.shape[0] > RANSAC_MIN_MATCHES
        matchesMask = None
        
        if do_ransac:
            logger.info("Filtering with RANSAC...")
            matched_pts_1 = kpts_1[match_idxs[:, 0]]
            matched_pts_2 = kpts_2[match_idxs[:, 1]]
            
            H, mask = KG.ransac.RANSAC(model_type="fundamental", inl_th=inl_th)(
                matched_pts_1, matched_pts_2
            )
            matchesMask = mask.int().ravel().tolist()
            inlier_count = mask.sum().item()
            log_text += f"RANSAC inliers: {inlier_count}/{mask.shape[0]} "
            log_text += f"({100*inlier_count/mask.shape[0]:.1f}%)\n"
        else:
            log_text += "Not enough matches for RANSAC\n"
        
        # Convert keypoints to OpenCV format
        kpts_1_cv = to_cv_kpts(kpts_1, score_1)
        kpts_2_cv = to_cv_kpts(kpts_2, score_2)
        
        # Draw keypoints
        logger.info("Generating visualizations...")
        keypoints_raw_1 = cv2.drawKeypoints(
            image1_np, kpts_1_cv, image1_np.copy(),
            color=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
        )
        keypoints_raw_2 = cv2.drawKeypoints(
            image2_np, kpts_2_cv, image2_np.copy(),
            color=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
        )
        
        # Pad and concatenate
        max_height = max(keypoints_raw_1.shape[0], keypoints_raw_2.shape[0])
        if keypoints_raw_1.shape[0] < max_height:
            pad_height = max_height - keypoints_raw_1.shape[0]
            keypoints_raw_1 = np.pad(
                keypoints_raw_1,
                ((0, pad_height), (0, 0), (0, 0)),
                mode='constant',
                constant_values=255
            )
        elif keypoints_raw_2.shape[0] < max_height:
            pad_height = max_height - keypoints_raw_2.shape[0]
            keypoints_raw_2 = np.pad(
                keypoints_raw_2,
                ((0, pad_height), (0, 0), (0, 0)),
                mode='constant',
                constant_values=255
            )
        
        keypoints_raw = np.concatenate((keypoints_raw_1, keypoints_raw_2), axis=1)
        
        # Draw all matches
        result_raw = cv2.drawMatches(
            image1_np, kpts_1_cv,
            image2_np, kpts_2_cv,
            cv2_matches, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        
        # Draw RANSAC filtered matches
        result_ransac = None
        if do_ransac:
            result_ransac = cv2.drawMatches(
                image1_np, kpts_1_cv,
                image2_np, kpts_2_cv,
                cv2_matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(0, 0, 255),
                matchesMask=matchesMask,
                flags=cv2.DrawMatchesFlags_DEFAULT,
            )
        
        # Convert to PIL
        result_raw_pil = Image.fromarray(cv2.cvtColor(result_raw, cv2.COLOR_BGR2RGB))
        result_ransac_pil = (
            Image.fromarray(cv2.cvtColor(result_ransac, cv2.COLOR_BGR2RGB))
            if result_ransac is not None else None
        )
        keypoints_pil = Image.fromarray(cv2.cvtColor(keypoints_raw, cv2.COLOR_BGR2RGB))
        
        logger.info("Extraction complete")
        return log_text, result_ransac_pil, result_raw_pil, keypoints_pil
    
    except Exception as e:
        logger.error(f"Error during extraction: {str(e)}")
        import traceback
        error_log = f"Error: {str(e)}\n{traceback.format_exc()}\n"
        return error_log, None, None, None


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

demo = gr.Interface(
    fn=extract_keypoints,
    inputs=[
        gr.Image(type="pil", label="Image 1"),
        gr.Image(type="pil", label="Image 2"),
        gr.Slider(
            minimum=0.1,
            maximum=3.0,
            step=0.1,
            value=0.5,
            label="RANSAC Inlier Threshold",
            info="Lower = stricter filtering, fewer but more reliable matches",
        ),
    ],
    outputs=[
        gr.Textbox(type="text", label="Log", max_lines=10),
        gr.Image(type="pil", label="Keypoint Matches (RANSAC Filtered)"),
        gr.Image(type="pil", label="All Keypoint Matches"),
        gr.Image(type="pil", label="Detected Keypoints"),
    ],
    title="RIPE: Robust Keypoint Extraction",
    description=DESCRIPTION_TEXT,
    examples=[
        [
            "assets/all_souls_000013.jpg",
            "assets/all_souls_000055.jpg",
            0.5,
        ],
    ],
    flagging_mode="never",
    theme="default",
    cache_examples=False,
)

if __name__ == "__main__":
    logger.info("Starting RIPE Gradio interface...")
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
    )