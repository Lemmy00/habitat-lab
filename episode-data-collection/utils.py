import cv2
import numpy as np
import torch
import shutil
import os
import math
import copy
import habitat.utils.geometry_utils as geo_utils
import imageio
import zipfile

import torch
import torchvision.transforms as T
import torchvision.models as models

from scipy import stats
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_to_list
from habitat.utils.visualizations import maps
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

from vint_based import load_distance_model
from vip import load_vip
from vint_train.models.vint.vint import ViNT


def zip_directory(src_dir: str, dst_zip: str) -> None:
    """Zip entire directory with *no* compression (ZIP_STORED)."""
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, start=src_dir)
                zf.write(abs_path, arcname=rel_path)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def quaternion_to_yaw(quat):
    """
    Extract yaw from a quaternion, assuming (x, y, z, w) order.
    """
    x, y, z, w = geo_utils.quaternion_to_list(quat)
    
    # Using standard formula: 
    #   yaw = atan2(2*(w*y + x*z), 1 - 2*(y^2 + z^2))
    sin_yaw = 2.0 * (w * y + x * z)
    cos_yaw = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return yaw


def angle_difference(a, b):
    """
    Returns the angle difference (a - b) in the range (-pi, pi].
    """
    diff = (a - b + np.pi) % (2.0 * np.pi) - np.pi
    return diff


def get_desired_yaw(goal_pos, current_pos):
    dx = goal_pos[0] - current_pos[0]
    dy = goal_pos[1] - current_pos[1]

    return np.arctan2(dy, dx)


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def average_min_distance(goal_feat: torch.Tensor, current_feat: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    Computes the average minimum distance between each feature in current_feat and the closest feature in goal_feat.

    Args:
        goal_feat (torch.Tensor): Tensor of shape (BS, NR_FEATURES, NR_CHANNELS)
        current_feat (torch.Tensor): Tensor of shape (BS, NR_FEATURES, NR_CHANNELS)
        k (int): Number of closest matches to consider (default: 1, meaning min distance)

    Returns:
        torch.Tensor: The average of the minimum distances for each feature.
    """
    # Compute pairwise Euclidean distances between current_feat and goal_feat
    diff = current_feat.unsqueeze(2) - goal_feat.unsqueeze(1)  # Shape: (BS, NR_FEATURES, NR_FEATURES, NR_CHANNELS)
    dist = torch.norm(diff, dim=-1)  # Shape: (BS, NR_FEATURES, NR_FEATURES)
    
    # Find the k closest goal features for each current feature
    min_distances, _ = torch.topk(dist, k, dim=-1, largest=False)
    
    # Compute the mean over the selected min distances
    avg_min_dist = min_distances.mean()
    
    return avg_min_dist


def rotate_agent_state(agent_state, turn_angle_degrees):
    """
    Return a new agent state with its rotation rotated by `turn_angle_degrees` around the up-axis.
    This example assumes that the up-axis is (0, 1, 0) and that agent_state.rotation is a quaternion.
    """
    # Convert turn angle to radians.
    turn_angle = math.radians(turn_angle_degrees)
    
    # Define up-axis vector.
    up_axis = np.array([0, 1, 0])
    
    # Compute the quaternion representing a rotation by turn_angle about up_axis.
    # One common approach is to compute the coefficients for a quaternion rotation:
    # q = [x, y, z, w] where (x,y,z) = axis * sin(theta/2) and w = cos(theta/2)
    sin_half_angle = math.sin(turn_angle / 2)
    cos_half_angle = math.cos(turn_angle / 2)
    q_coeffs = [up_axis[0] * sin_half_angle, up_axis[1] * sin_half_angle, up_axis[2] * sin_half_angle, cos_half_angle]
    
    # Create a quaternion from coefficients using habitat's helper.
    turn_quat = quaternion_from_coeff(q_coeffs)
    
    # Multiply the turn quaternion with the agent's current rotation.
    # Note: Depending on the quaternion library, the order of multiplication matters.
    new_rotation = turn_quat * agent_state.rotation

    # Create a deep copy of the agent state to avoid mutating the original.
    new_agent_state = copy.deepcopy(agent_state)
    new_agent_state.rotation = new_rotation
    return new_agent_state


def get_default_transform(normalize=True):
    transforms = [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    if normalize:
        transforms.append(T.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def load_embedding(rep: str):
    """
    Dynamically load a model and its transform based on the representation name.
    Model IDs must match rep exactly for decoders (e.g. 'one_scene_decoder_20').
    """
    # VIP-based models
    if rep.startswith('vip-'):
        model = load_vip(modelid=rep)
        transform = get_default_transform()
    # VIP original model
    elif rep == 'vip':
        model = load_vip(modelid='resnet50')
        transform = get_default_transform(normalize=False)
    # ViNT original model
    elif rep == "vint":
        model = ViNT(obs_encoder='efficientnet-b0', mha_num_attention_layers=4)
        model.load_state_dict(torch.load("/cluster/home/lmilikic/.dist_models/vint/vint.pt"))
        model = model.to('cuda')
        transform = get_default_transform()
    # Distance decoders, Vint distance, and one-scene quasi MSE
    elif rep.startswith(('one_scene_decoder', 'dist_decoder', 'vint_dist', 'one_scene_quasi', 'quasi')):
        model = load_distance_model(modelid=rep)
        transform = get_default_transform()
    # Standard ResNet
    elif rep == 'resnet':
        model = models.resnet50(pretrained=True, progress=False)
        transform = get_default_transform()
    # DINO-V2 ViT-B/14
    elif rep == 'dino':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        transform = get_default_transform()
    else:
        raise ValueError(f"Unsupported representation: '{rep}'")
    return model, transform


def compute_correlation(distances, gt, method: str = 'kendall'):
    if method == 'spearman':
        corr = stats.spearmanr(distances, gt)
        symbol = '$S_\\rho$'
    elif method == 'kendall':
        corr = stats.kendalltau(distances, gt)
        symbol = '$\\tau$'
    else:
        raise ValueError(f"Unsupported correlation: {method}")
    return corr.statistic, corr.pvalue, symbol


def images_to_video(
    images: list,
    output_path: str,
    fps: int = 10,
    quality: float = 5.0,
    verbose: bool = True,
    **kwargs,
):
    """
    Saves a list of RGB images to a video file using imageio/FFMPEG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        quality=quality,
        **kwargs,
    )
    iterator = images if not verbose else tqdm(images, desc='Writing frames')
    for frame in iterator:
        writer.append_data(frame)
    writer.close()


def animate_episode(
    frames: list,
    maps: list,
    geo_distances: list,
    pred_distances: list,
    goal_frame: np.ndarray,
    out_video: str,
    max_len: int,
    fps: int,
    corr_method: str = 'kendall',
    pause_sec: float = 0.0
):
    """
    Renders an episode sequence (trajectory, top-down map, distances, goal) into a video file.
    """
    # Truncate sequences
    frames = frames[-(max_len+1):]
    maps   = maps[-(max_len+1):]
    geo    = np.array(geo_distances[-(max_len+1):],  dtype=float)
    pred   = np.array(pred_distances[-(max_len+1):], dtype=float)
    baseline = np.linspace(pred[0], 0.0, len(frames))

    # Correlations
    τ_geo_base, _, sym = compute_correlation(geo,  baseline, corr_method)
    τ_pred_base, _, _   = compute_correlation(pred, baseline, corr_method)
    τ_pred_geo,  _, _   = compute_correlation(pred, geo,      corr_method)

    # Calculate pause frames
    pause_frames = int(fps * pause_sec)
    total_steps = len(frames) + pause_frames

    # Prepare rendering
    rendered = []
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    ax_curve, ax_video, ax_map, ax_goal = axes
    max_y = max(geo.max(), pred.max(), baseline.max())

    def render_step(i):
        idx = min(i, len(frames)-1)
        x = np.arange(idx + 1)

        # Distances curves
        ax_curve.clear()
        ax_curve.plot(x, geo[:idx+1], label=f"Geo ({sym}: {τ_geo_base:.2f})", linewidth=2)
        ax_curve.plot(x, pred[:idx+1], '-.',
                      label=f"Pred ({sym}: {τ_pred_base:.2f}, vsGeo: {τ_pred_geo:.2f})", linewidth=2)
        ax_curve.plot(x, baseline[:idx+1], '--', label="Baseline", linewidth=2)
        ax_curve.set(xlim=(0, len(frames)-1), ylim=(0, max_y),
                     xlabel="Frame", ylabel="Distance",
                     title="Geo & Pred vs Baseline")
        ax_curve.legend(loc="upper right", fontsize="x-small")

        # RGB trajectory
        ax_video.clear()
        ax_video.imshow(frames[idx])
        ax_video.axis("off")
        ax_video.set_title("Trajectory")

        # Top-down map
        ax_map.clear()
        ax_map.imshow(maps[idx])
        ax_map.axis("off")
        ax_map.set_title("Top‑down Map")

        # Goal image
        ax_goal.clear()
        ax_goal.imshow(goal_frame)
        ax_goal.axis("off")
        ax_goal.set_title("Goal Image")

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        rendered.append(buf.reshape(height, width, 3))

    # Render all frames
    for i in range(total_steps):
        render_step(i)
    plt.close(fig)

    # Save video
    images_to_video(rendered, out_video, fps=fps)
    print(f"Saved animation to {out_video}")


class ImageDescriber:
    def __init__(self, device=None, prompt_template="the {label} is"):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.prompt_template = prompt_template
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def describe_image(self, image, label, max_length=150):
        prompt = self.prompt_template.format(label=label)
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        output = self.model.generate(**inputs, max_length=max_length)
        
        # Full generated tokens including prompt and new text
        generated_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Remove prompt fragment from generated_text safely
        prompt = prompt.lower()
        if prompt in generated_text:
            generated_text = generated_text.split(prompt, 1)[-1].strip()
        
        return generated_text
