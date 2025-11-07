# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

import os
import cv2
import sys
import gc
import glob
import time
import copy
import shutil
import trimesh
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from datetime import datetime

# ---------------- VGGT / COLMAP deps ----------------
sys.path.append("vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
import pycolmap

# ---------------- Determinism / Memory knobs ----------------
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False   # avoid huge workspaces
torch.backends.cudnn.deterministic = False

# ============================================================
#                         GLOBAL MODEL
# ============================================================
print("Initializing and loading VGGT model...")
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location=_device))
_model.eval().to(_device)

# ============================================================
#                      GPU/CPU Garbage Collect
# ============================================================
def gpu_gc(tag=""):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[GPU_GC]{' '+tag if tag else ''} free={free/1024**2:.1f} MiB / total={total/1024**2:.1f} MiB")

# ============================================================
#                VGGT forward → cams + depth @518
# ============================================================
@torch.no_grad()
def _run_vggt_cams_depth_518(model, images_518, dtype):
    """
    images_518: [S,3,518,518] on CUDA
    returns numpy arrays on CPU; nothing GPU-resident is returned.
    """
    with torch.cuda.amp.autocast(dtype=dtype):
        batched = images_518[None]  # [1,S,3,518,518]
        aggregated_tokens_list, ps_idx = model.aggregator(batched)
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_518.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batched, ps_idx)

    # Move *results* to CPU numpy
    extrinsic = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).detach().cpu().numpy()
    depth_map = depth_map.squeeze(0).detach().cpu().numpy()
    depth_conf = depth_conf.squeeze(0).detach().cpu().numpy()

    # Proactively drop intermediates
    try:
        del aggregated_tokens_list, ps_idx, pose_enc
    except Exception:
        pass

    return extrinsic, intrinsic, depth_map, depth_conf

# ============================================================
#           Rename/rescale COLMAP to original resolution
# ============================================================
def _rename_rescale_colmap(
    reconstruction, base_image_names, original_coords_np, img_size, shared_camera=False, shift_p2d=True
):
    rescale_camera = True
    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = base_image_names[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords_np[pyimageid - 1, -2:]  # (W,H)
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2.0
            pred_params[-2:] = real_pp  # cx,cy center
            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_p2d:
            top_left = original_coords_np[pyimageid - 1, :2]
            for p2D in pyimage.points2D:
                p2D.xy = (p2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False
    return reconstruction

# ============================================================
#                Export sparse points to GLB
# ============================================================
def _export_colmap_points_glb(reconstruction, glb_path):
    if len(reconstruction.points3D) == 0:
        raise ValueError("No 3D points in reconstruction after BA.")
    xyz, rgb = [], []
    for _, pt in reconstruction.points3D.items():
        xyz.append(pt.xyz)
        rgb.append(pt.color)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    trimesh.Scene(trimesh.PointCloud(xyz, colors=rgb)).export(glb_path)

# ============================================================
#      MAIN: VGGT → unproject depth → COLMAP BA (NO TRACKS)
# ============================================================
def vggt_ba_reconstruct_to_glb_no_tracks(
    target_dir,
    *,
    conf_thres_value=5.0,          # depth confidence threshold
    max_points_for_colmap=120000,  # random cap across all frames (reduce if RAM heavy)
    shared_camera=False,           # True: single intrinsics across images
    camera_type="PINHOLE",         # SIMPLE_PINHOLE / PINHOLE / ...
    vggt_fixed_resolution=518,
    img_load_resolution=1024
):
    assert torch.cuda.is_available(), "CUDA is required for VGGT forward pass."

    image_dir = os.path.join(target_dir, "images")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_names = [os.path.basename(p) for p in image_paths]

    # 1) Load high-res (for bookkeeping/colors)
    images_hi, original_coords = load_and_preprocess_images_square(image_paths, img_load_resolution)
    images_hi = images_hi.to("cuda", non_blocking=True)  # [S,3,H,W]
    original_coords = original_coords.to("cuda", non_blocking=True)

    # 2) Downscale copy for VGGT forward at 518
    images_518 = F.interpolate(
        images_hi, size=(vggt_fixed_resolution, vggt_fixed_resolution),
        mode="bilinear", align_corners=False
    )

    # 3) VGGT forward on CUDA → numpy on CPU
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    _model_local = _model  # alias
    extrinsic, intrinsic, depth_map, depth_conf = _run_vggt_cams_depth_518(_model_local, images_518, dtype)

    # 4) Unproject depth (CPU numpy)
    points_3d_full = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)  # (S,H,W,3)

    # 5) Colors at 518 (uint8), CPU numpy
    points_rgb = F.interpolate(
        images_hi, size=(vggt_fixed_resolution, vggt_fixed_resolution),
        mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.detach().cpu().numpy() * 255).astype(np.uint8)   # (S,3,H,W)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)                              # (S,H,W,3)

    # 6) Build (x,y,frame) grid
    S, H, W, _ = points_3d_full.shape
    points_xyf = create_pixel_coordinate_grid(S, H, W)   # (S*H*W, 3) in (x,y,f)

    # 7) Confidence mask + random cap
    conf_mask = (depth_conf >= conf_thres_value)  # (S,H,W,1) or (S,H,W)
    conf_mask = np.asarray(conf_mask).squeeze(-1) if conf_mask.ndim == 4 else np.asarray(conf_mask)
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    # 8) Apply mask (CPU arrays)
    points_3d = points_3d_full[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    # ------- HARD CLEAN VGGT FROM GPU BEFORE BA -------
    try:
        # free CUDA tensors kept so far
        del images_518
    except Exception:
        pass
    try:
        # offload global model from GPU (no need for it beyond this point)
        _model_local.to("cpu")
    except Exception:
        pass
    try:
        # free hi-res images too; BA runs on CPU with numpy/pycolmap
        del images_hi
    except Exception:
        pass
    gpu_gc("no-tracks-before-colmap")

    # 9) Build COLMAP reconstruction without tracks (CPU)
    image_size_518 = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size_518,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )
    if reconstruction is None:
        raise ValueError("COLMAP reconstruction init failed (no-tracks).")

    # 10) Bundle Adjustment (CPU)
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    # 11) Rescale/rename to original geometry
    reconstruction = _rename_rescale_colmap(
        reconstruction,
        base_image_names=base_names,
        original_coords_np=original_coords.detach().cpu().numpy(),
        img_size=vggt_fixed_resolution,
        shared_camera=shared_camera,
        shift_p2d=True,
    )

    # 12) Save outputs
    sparse_dir = os.path.join(target_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)

    if len(reconstruction.points3D) > 0:
        ply_xyz, ply_rgb = [], []
        for _, pt in reconstruction.points3D.items():
            ply_xyz.append(pt.xyz); ply_rgb.append(pt.color)
        ply_xyz = np.asarray(ply_xyz, dtype=np.float32)
        ply_rgb = np.asarray(ply_rgb, dtype=np.uint8)
        trimesh.PointCloud(ply_xyz, colors=ply_rgb).export(os.path.join(sparse_dir, "points.ply"))

    glb_path = os.path.join(target_dir, "ba_scene.glb")
    _export_colmap_points_glb(reconstruction, glb_path)

    # 13) Final clean
    try:
        del reconstruction, points_3d_full, points_3d, points_xyf, points_rgb, depth_map, depth_conf, extrinsic, intrinsic
    except Exception:
        pass
    gpu_gc("no-tracks-after-BA")

    # Return the BA result path and frame count
    return glb_path, len(image_paths)

# ============================================================
#                 Upload handlers (unchanged)
# ============================================================
def handle_uploads(input_video, input_images):
    start_time = time.time()
    gc.collect()
    gpu_gc("before-upload")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # images
    if input_images is not None:
        for file_data in input_images:
            file_path = file_data["name"] if isinstance(file_data, dict) and "name" in file_data else file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # video → 1 fps frames
    if input_video is not None:
        video_path = input_video["name"] if isinstance(input_video, dict) and "name" in input_video else input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(max(1, fps) * 1)
        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    image_paths = sorted(image_paths)
    print(f"Files copied to {target_dir_images}; took {time.time() - start_time:.3f} seconds")
    gpu_gc("after-upload")
    return target_dir, image_paths

def update_gallery_on_upload(input_video, input_images):
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."

# ============================================================
#                Gradio: BA-only reconstruction
# ============================================================
def clear_fields():
    return None

def update_log():
    return "Loading and Reconstructing..."

def gradio_demo_colmap(
    target_dir,
    conf_thres=5.0,                 # used for BA point selection (depth confidence)
    frame_filter="All",             # kept for UI compatibility; unused in BA path
    mask_black_bg=False,            # UI compatibility
    mask_white_bg=False,            # UI compatibility
    show_cam=True,                  # UI compatibility
    mask_sky=False,                 # UI compatibility
    prediction_mode="VGGT + BA",    # label
):
    """
    BA-only flow:
      - Run VGGT (cams+depth) @518
      - Unproject depth → sparse 3D
      - Build COLMAP reconstruction (no tracks)
      - Run BA (CPU)
      - Export GLB for viewer
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None

    start_time = time.time()
    gc.collect()
    gpu_gc("start-of-reconstruct")

    # Prepare dropdown (just to show something consistent)
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    ba_glb = None
    ba_ok = False
    ba_log = ""
    try:
        ba_glb, n_frames = vggt_ba_reconstruct_to_glb_no_tracks(
            target_dir,
            conf_thres_value=float(conf_thres),
            max_points_for_colmap=120000,
            shared_camera=False,
            camera_type="PINHOLE",
            vggt_fixed_resolution=518,
            img_load_resolution=1024
        )
        ba_ok = True
    except Exception as e:
        ba_log = f" (error: {str(e)[:200]})"
        gpu_gc("after-BA-fail")

    end_time = time.time()
    gpu_gc("end-of-reconstruct")

    if ba_ok and ba_glb and os.path.exists(ba_glb):
        log_msg = f"Reconstruction complete. VGGT + BA saved to ./sparse and ba_scene.glb{ba_log}. Time: {end_time - start_time:.2f}s"
        return ba_glb, log_msg, gr.Dropdown(choices=frame_filter_choices, value="All", interactive=True)
    else:
        return None, f"Reconstruction failed{ba_log}. Check logs.", gr.Dropdown(choices=frame_filter_choices, value="All", interactive=True)

def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."
    ba_glb = os.path.join(target_dir, "ba_scene.glb")
    if os.path.exists(ba_glb):
        return ba_glb, "Updating Visualization (VGGT + BA)"
    return None, "No reconstruction available. Please run 'Reconstruct' first."

# ============================================================
#                     Gradio UI (no examples)
# ============================================================
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
    theme=theme,
    css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    #my_radio .wrap { display: flex; flex-wrap: nowrap; justify-content: center; align-items: center; }
    #my_radio .wrap label { display: flex; width: 50%; justify-content: center; align-items: center; margin: 0; padding: 10px 0; box-sizing: border-box; }
    """,
) as demo:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT → COLMAP (Bundle Adjustment)</h1>
    <p><a href="https://github.com/facebookresearch/vggt">🐙 VGGT GitHub</a></p>
    <div style="font-size: 16px; line-height: 1.5;">
      <p>Upload several images of a static scene (spatial spread). The app estimates camera poses and depth with VGGT (on GPU), unprojects to 3D, and refines everything with COLMAP Bundle Adjustment (on CPU). The viewer shows the BA result.</p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video (optional)", interactive=True)
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)
            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (VGGT → BA)**")
                log_output = gr.Markdown(
                    "Please upload a set of images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct (VGGT + BA)", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["VGGT + BA"],
                    label="Pipeline",
                    value="VGGT + BA",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=20, value=5.0, step=0.5,
                                       label="Depth Confidence Threshold (for BA points)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="(unused) Frame Filter")
                with gr.Column():
                    show_cam = gr.Checkbox(label="(unused) Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="(unused) Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="(unused) Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="(unused) Filter White Background", value=False)

    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo_colmap,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]
    )

    # live re-viz: just reload ba_scene.glb if exists
    for comp in (conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode):
        comp.change(
            update_visualization,
            [
                target_dir_output,
                conf_thres,
                frame_filter,
                mask_black_bg,
                mask_white_bg,
                show_cam,
                mask_sky,
                prediction_mode,
                is_example,
            ],
            [reconstruction_output, log_output],
        )

    # auto-preview on upload
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)
