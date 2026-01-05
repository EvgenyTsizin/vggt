#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gc
import cv2
import time
import json
import glob
import shutil
import numpy as np
import torch
import trimesh
import open3d as o3d
import gradio as gr
from datetime import datetime

sys.path.append("vggt/")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1. CONSISTENCY FILTER (Legacy Shape Logic)
# -------------------------------------------------------------------------
def filter_predictions_consistency(predictions, depth_limit=0.8, min_views=3, geo_tol=0.1):
    D = predictions.get("depth")
    WP = predictions.get("world_points_from_depth")
    E = predictions.get("extrinsic")
    K = predictions.get("intrinsic")

    if D is None or WP is None or E is None or K is None:
        return predictions

    # --- LEGACY SHAPE HANDLING ---
    WP = np.array(WP)
    if D is not None:
        D = np.array(D)
        if D.ndim == 4 and D.shape[-1] == 1:
            D = D[..., 0]
        D = D.astype(np.float32, copy=False)
    
    S, H, W = WP.shape[:3]
    # -----------------------------

    if E.shape[1] == 3:
        last_row = np.array([0, 0, 0, 1]).reshape(1, 1, 4).repeat(S, axis=0)
        E_mat = np.concatenate([E, last_row], axis=1)
    else:
        E_mat = E

    Rs = E_mat[:, :3, :3]
    ts = E_mat[:, :3, 3]

    final_mask = np.zeros((S, H, W), dtype=bool)

    for i in range(S):
        mask_i = (D[i] < depth_limit)
        if not np.any(mask_i): continue

        pts_w = WP[i][mask_i]
        counts = np.ones(pts_w.shape[0], dtype=np.int32)

        for j in range(S):
            if i == j: continue
            
            P_cj = pts_w @ Rs[j].T + ts[j]
            valid_z = P_cj[:, 2] > 0.001
            
            fx, fy = K[j, 0, 0], K[j, 1, 1]
            cx, cy = K[j, 0, 2], K[j, 1, 2]
            
            u = (P_cj[:, 0] * fx / P_cj[:, 2]) + cx
            v = (P_cj[:, 1] * fy / P_cj[:, 2]) + cy

            in_bounds = (u >= 0) & (u < W - 0.5) & (v >= 0) & (v < H - 0.5)
            valid_proj = valid_z & in_bounds

            if np.any(valid_proj):
                idx_valid = np.where(valid_proj)[0]
                u_valid = np.round(u[idx_valid]).astype(int)
                v_valid = np.round(v[idx_valid]).astype(int)

                d_sample = D[j, v_valid, u_valid]
                z_calc   = P_cj[idx_valid, 2]

                cond_depth = d_sample < depth_limit
                diff = np.abs(d_sample - z_calc)
                cond_geo = diff < (geo_tol * z_calc + 0.05) 

                match = cond_depth & cond_geo
                counts[idx_valid[match]] += 1

        keep_indices = counts >= min_views
        flat_indices = np.flatnonzero(mask_i)
        final_indices = flat_indices[keep_indices]
        
        frame_mask = np.zeros(H * W, dtype=bool)
        frame_mask[final_indices] = True
        final_mask[i] = frame_mask.reshape(H, W)

    if predictions["depth"].ndim == 4:
         predictions["depth"][~final_mask] = np.nan
    else:
         predictions["depth"][~final_mask] = np.nan

    predictions["world_points_from_depth"][~final_mask] = np.nan
    return predictions


# -------------------------------------------------------------------------
# 2. SMOOTH MESH GENERATION (SOR + Poisson)
# -------------------------------------------------------------------------
def generate_smooth_mesh(predictions, target_dir):
    print("Generating Smooth Mesh (SOR -> Poisson)...")
    
    WP = predictions.get("world_points_from_depth")
    S, H, W = WP.shape[:3]
    
    # Load Colors
    images_dir = os.path.join(target_dir, "images")
    img_files = sorted(os.listdir(images_dir)) if os.path.isdir(images_dir) else []
    rgb_stack = []
    
    for i in range(S):
        p = os.path.join(images_dir, img_files[i])
        im_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if im_bgr is None: im_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        else: im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        
        if im_rgb.shape[:2] != (H, W):
            im_rgb = cv2.resize(im_rgb, (W, H))
        rgb_stack.append(im_rgb)
    
    # Extract Points
    all_xyz = []
    all_rgb = []
    
    for i in range(S):
        xyz_frame = WP[i].reshape(-1, 3)
        rgb_frame = rgb_stack[i].reshape(-1, 3)
        mask = np.isfinite(xyz_frame).all(axis=1)
        if np.any(mask):
            all_xyz.append(xyz_frame[mask])
            all_rgb.append(rgb_frame[mask])

    if not all_xyz:
        print("Error: No points found.")
        return None

    pts = np.concatenate(all_xyz, axis=0).astype(np.float64)
    colors = np.concatenate(all_rgb, axis=0).astype(np.float64) / 255.0

    # Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"Raw Points: {len(pcd.points)}")

    # --- ALGORITHM START ---

    # A. Statistical Outlier Removal (SOR)
    # This removes the "fuzz" (points floating slightly off the layers)
    print("Removing Outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    print(f"Points after Clean: {len(pcd.points)}")

    # B. Normal Estimation
    # Critical for Poisson. We use a larger radius to 'smooth' local direction.
    print("Estimating Normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # C. Poisson Surface Reconstruction
    # This solves the implicit surface function, effectively fusing/averaging layers 
    # into a single watertight mesh.
    print("Running Poisson Reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )

    # D. Cleanup Mesh
    # Remove vertices with low support density (ghost geometry)
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # E. Color Projection (KD-Tree)
    # Project colors from the Clean PCD back to the Mesh Vertices
    print("Projecting Colors to Mesh...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = []
    mesh_verts = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    
    for i in range(len(mesh_verts)):
        # Find 1 nearest neighbor in original cloud
        _, idx, _ = pcd_tree.search_knn_vector_3d(mesh_verts[i], 1)
        mesh_colors.append(pcd_colors[idx[0]])
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh_colors))

    # Convert to Trimesh for export
    print("Converting to Trimesh...")
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors)
    )
    
    return trimesh.Scene([tri_mesh])


# -------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------------
def save_results(target_dir, predictions):
    # Minimal export logic for brevity, focusing on the mesh
    pass # Data is already in memory for the GLB export

def run_model(target_dir):
    print(f"Processing: {target_dir}")
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    images = load_and_preprocess_images(image_names).to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred = model(images)
    
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    pred["extrinsic"] = ext
    pred["intrinsic"] = intr
    
    # Detach to numpy
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            pred[k] = v.detach().cpu().numpy().squeeze(0) if v.ndim > 0 and v.shape[0]==1 else v.detach().cpu().numpy()
            
    # Unproject
    try:
        pred["world_points_from_depth"] = unproject_depth_map_to_point_map(
            pred["depth"], pred["extrinsic"], pred["intrinsic"]
        )
    except:
        pred["world_points_from_depth"] = None
        
    return pred

def handle_upload(vid, imgs):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    td = f"input_{ts}"
    td_img = os.path.join(td, "images")
    os.makedirs(td_img, exist_ok=True)
    
    if imgs:
        for f in imgs:
            shutil.copy(f.name if hasattr(f, 'name') else f, os.path.join(td_img, os.path.basename(f.name if hasattr(f, 'name') else f)))
    elif vid:
        cap = cv2.VideoCapture(vid)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps)) # 1 FPS
        cnt = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if cnt % interval == 0:
                cv2.imwrite(os.path.join(td_img, f"{saved:05d}.png"), frame)
                saved += 1
            cnt += 1
    return td

# -------------------------------------------------------------------------
# 4. SIMPLIFIED GRADIO UI
# -------------------------------------------------------------------------
def process_pipeline(target_dir):
    if not target_dir: return None, "Upload first."
    
    # A. Inference
    preds = run_model(target_dir)
    
    # B. Filter (Depth < 0.8, >= 3 Cams)
    preds = filter_predictions_consistency(preds, depth_limit=0.8, min_views=3)
    
    # C. Smooth Mesh (SOR + Poisson)
    scene = generate_smooth_mesh(preds, target_dir)
    
    if scene is None:
        return None, "Reconstruction Failed (No points left)."
        
    out_path = os.path.join(target_dir, "fused_mesh.glb")
    scene.export(file_obj=out_path)
    
    return out_path, "Done. Mesh created."


with gr.Blocks() as demo:
    gr.Markdown("# VGGT Mesh Reconstruction")
    gr.Markdown("Uses **Statistical Outlier Removal** & **Poisson Reconstruction** to smooth layers.")
    
    t_dir = gr.State()
    
    with gr.Row():
        with gr.Column():
            in_vid = gr.Video(label="Input Video")
            in_imgs = gr.File(label="Input Images", file_count="multiple")
            btn_upload = gr.Button("1. Upload & Prepare")
            btn_run = gr.Button("2. Reconstruct Mesh", variant="primary")
            log = gr.Textbox(label="Status")
            
        with gr.Column():
            out_3d = gr.Model3D(label="Fused Mesh Result", height=600)
            
    def upload(v, i):
        td = handle_upload(v, i)
        return td, f"Ready: {td}"

    btn_upload.click(upload, [in_vid, in_imgs], [t_dir, log])
    btn_run.click(process_pipeline, [t_dir], [out_3d, log])

demo.launch(share=True)