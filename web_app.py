import gradio as gr
import torch
import cv2
import numpy as np
import kornia.feature as KF
import kornia.geometry as KG
from torchvision.io import decode_image

from ripe import vgg_hyper
from ripe.utils.utils import (
    resize_image,
    to_cv_kpts,
    cv2_matches_from_kornia,
)

# -------------------------
# Model loading (once)
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = vgg_hyper().to(device)
model.eval()


def ripe_match(img1, img2):
    if img1 is None or img2 is None:
        return None

    # Convert Gradio images to torch tensors
    image1 = torch.from_numpy(img1).permute(2, 0, 1).float().to(device) / 255.0
    image2 = torch.from_numpy(img2).permute(2, 0, 1).float().to(device) / 255.0

    image1 = resize_image(image1)
    image2 = resize_image(image2)

    kpts1, desc1, score1 = model.detectAndCompute(
        image1, threshold=0.5, top_k=2048
    )
    kpts2, desc2, score2 = model.detectAndCompute(
        image2, threshold=0.5, top_k=2048
    )

    matcher = KF.DescriptorMatcher("mnn")
    match_dists, match_idxs = matcher(desc1, desc2)

    if match_idxs.numel() == 0:
        return None

    pts1 = kpts1[match_idxs[:, 0]]
    pts2 = kpts2[match_idxs[:, 1]]

    _, mask = KG.ransac.RANSAC(
        model_type="fundamental", inl_th=1.0
    )(pts1, pts2)

    matchesMask = mask.int().ravel().tolist()

    img1_cv = (image1.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img2_cv = (image2.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    result = cv2.drawMatches(
        img1_cv,
        to_cv_kpts(kpts1, score1),
        img2_cv,
        to_cv_kpts(kpts2, score2),
        cv2_matches_from_kornia(match_dists, match_idxs),
        None,
        matchColor=(0, 255, 0),
        matchesMask=matchesMask,
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    return result


# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks(title="RIPE – Keypoint Matching") as demo:
    gr.Markdown(
        """
        # 🔑 RIPE – Robust Keypoint Extraction  
        Upload **two images of the same scene** and see RIPE match keypoints.
        """
    )

    with gr.Row():
        img1 = gr.Image(label="Image 1", type="numpy")
        img2 = gr.Image(label="Image 2", type="numpy")

    btn = gr.Button("Match Images 🚀")
    output = gr.Image(label="Matched Keypoints")

    btn.click(fn=ripe_match, inputs=[img1, img2], outputs=output)

demo.launch()
