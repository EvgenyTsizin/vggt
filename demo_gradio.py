#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Full Gradio app with "save everything & zip" export (no images).
# Saves: cameras (npz/txt/json), depth maps + conf (npy), points (ply/npy),
# full results (npz), summary (json), and a downloadable ZIP.
# Also prints/shows the export folder in the UI.

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
import gradio as gr
from datetime import datetime
from pathlib import Path

sys.path.append("vggt/")

from visual_util import predictions_to_glb
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
# Exporter: save all requested artifacts and zip them (no images)
# -------------------------------------------------------------------------
def save_and_zip_results(target_dir: str, predictions: dict, conf_thres_percent: float = 50.0):
    """
    Save full reconstruction results (excluding images) into <target_dir>/export/
    and zip them. Returns (export_dir, zip_path).

    Saved content:
      export/
        cameras/
          cameras.npz            # intrinsic, extrinsic
          camera_params.txt      # human-readable
          cameras.json           # JSON copy (quick view)
        depth/
          depth_XXXX.npy         # per-frame depth (float32)
          conf_XXXX.npy          # per-frame depth confidence (float32)
          depth_meta.json        # shape info + stats
        points/
          points.ply             # point cloud (geometry-only)
          points.npy             # Nx3 numpy array (float32)
        meta/
          results.npz            # full predictions (intrinsic, extrinsic, depth, depth_conf, world_points_from_depth)
          summary.json           # sizes, counts, thresholds used
    """
    target_dir = str(target_dir)
    export_dir = os.path.join(target_dir, "export")
    cams_dir   = os.path.join(export_dir, "cameras")
    depth_dir  = os.path.join(export_dir, "depth")
    points_dir = os.path.join(export_dir, "points")
    meta_dir   = os.path.join(export_dir, "meta")
    for d in (cams_dir, depth_dir, points_dir, meta_dir):
        os.makedirs(d, exist_ok=True)

    # --- grab arrays safely ---
    K = predictions.get("intrinsic", None)
    E = predictions.get("extrinsic", None)
    depth = predictions.get("depth", None)                  # [S,H,W] or [S,H,W,1]
    depth_conf = predictions.get("depth_conf", None)        # [S,H,W]
    wp = predictions.get("world_points_from_depth", None)   # [S,H,W,3] if present

    # --- normalize shapes ---
    if depth is not None:
        depth = np.array(depth)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        depth = depth.astype(np.float32, copy=False)

    if depth_conf is not None:
        depth_conf = np.array(depth_conf).astype(np.float32, copy=False)

    # --- cameras: save npz, txt, json ---
    cam_npz = os.path.join(cams_dir, "cameras.npz")
    np.savez(cam_npz, intrinsic=np.array(K), extrinsic=np.array(E))

    cam_txt = os.path.join(cams_dir, "camera_params.txt")
    with open(cam_txt, "w") as f:
        f.write("=== Intrinsic (K) ===\n")
        f.write(repr(np.array(K)) + "\n\n")
        f.write("=== Extrinsic (camera-from-world) ===\n")
        f.write(repr(np.array(E)) + "\n")

    cam_json = os.path.join(cams_dir, "cameras.json")
    with open(cam_json, "w") as f:
        json.dump({
            "intrinsic": np.array(K).tolist() if K is not None else None,
            "extrinsic": np.array(E).tolist() if E is not None else None
        }, f, indent=2)

    # --- depth: save per-frame .npy + meta ---
    if depth is not None:
        S = depth.shape[0]
        for i in range(S):
            np.save(os.path.join(depth_dir, f"depth_{i:04d}.npy"), depth[i])
        depth_min = float(np.nanmin(depth)) if depth.size else None
        depth_max = float(np.nanmax(depth)) if depth.size else None
        depth_mean = float(np.nanmean(depth)) if depth.size else None

        if depth_conf is not None and depth_conf.shape[:3] == depth.shape[:3]:
            for i in range(S):
                np.save(os.path.join(depth_dir, f"conf_{i:04d}.npy"), depth_conf[i])

        with open(os.path.join(depth_dir, "depth_meta.json"), "w") as f:
            json.dump({
                "frames": int(S),
                "shape_per_frame": list(depth.shape[1:]),
                "stats": {"min": depth_min, "max": depth_max, "mean": depth_mean}
            }, f, indent=2)

    # --- points: build from world_points_from_depth if present ---
    pts_npy_path = os.path.join(points_dir, "points.npy")
    pts_ply_path = os.path.join(points_dir, "points.ply")
    saved_points = False
    if wp is not None:
        wp = np.array(wp)
        pts = wp.reshape(-1, 3)
        mask = np.isfinite(pts).all(axis=1)
        if depth_conf is not None and depth is not None:
            conf = depth_conf.reshape(-1)
            valid_conf = conf[np.isfinite(conf)]
            if valid_conf.size > 0:
                th = np.percentile(valid_conf, conf_thres_percent)
                mask &= (conf >= th)
        pts = pts[mask]
        # store numpy
        np.save(pts_npy_path, pts.astype(np.float32, copy=False))
        # write PLY
        try:
            import trimesh
            trimesh.PointCloud(pts.astype(np.float32, copy=False)).export(pts_ply_path)
            saved_points = True
        except Exception:
            # simple ASCII PLY fallback
            try:
                with open(pts_ply_path, "w") as f:
                    f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(pts.shape[0]))
                    f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                    for p in pts:
                        f.write(f"{p[0]} {p[1]} {p[2]}\n")
                saved_points = True
            except Exception:
                saved_points = False

    # --- meta: full NPZ & summary ---
    results_npz = os.path.join(meta_dir, "results.npz")
    np.savez_compressed(
        results_npz,
        intrinsic=np.array(K) if K is not None else None,
        extrinsic=np.array(E) if E is not None else None,
        depth=depth if depth is not None else None,
        depth_conf=depth_conf if depth_conf is not None else None,
        world_points_from_depth=np.array(wp) if wp is not None else None
    )

    summary = {
        "export_dir": export_dir,
        "saved": {
            "cameras_npz": cam_npz,
            "camera_params_txt": cam_txt,
            "cameras_json": cam_json,
            "depth_dir": depth_dir if depth is not None else None,
            "points_ply": pts_ply_path if saved_points else None,
            "points_npy": pts_npy_path if os.path.exists(pts_npy_path) else None,
            "results_npz": results_npz
        },
        "thresholds": {
            "conf_thres_percent": conf_thres_percent
        }
    }
    with open(os.path.join(meta_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- zip it ---
    zip_base = os.path.join(target_dir, os.path.basename(target_dir) + "_export")
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=export_dir)
    return export_dir, zip_path


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    print(f"Processing images from {target_dir}")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    model = model.to(device)
    model.eval()

    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            print("Running inference...")
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # tensors -> numpy
    for key in list(predictions.keys()):
        v = predictions[key]
        if isinstance(v, torch.Tensor):
            try:
                arr = v.detach().cpu().numpy()
                if arr.ndim > 0 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                predictions[key] = arr
            except Exception:
                predictions[key] = v

    # make world points from depth (best-effort)
    try:
        print("Computing world points from depth map...")
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
    except Exception as e:
        print("Warning: failed to compute world points from depth:", e)
        predictions["world_points_from_depth"] = None

    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) Handle uploaded video/images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

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
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # video -> frames (1 fps)
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video
        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps * 1))
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

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=50.0,  # percent
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Depthmap and Camera Branch",
):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save a lightweight predictions bundle (without images)
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez_compressed(
        prediction_save_path,
        pose_enc=predictions.get("pose_enc", None),
        depth=predictions.get("depth", None),
        depth_conf=predictions.get("depth_conf", None),
        world_points=predictions.get("world_points", None),
        world_points_conf=predictions.get("world_points_conf", None),
        extrinsic=predictions.get("extrinsic", None),
        intrinsic=predictions.get("intrinsic", None),
        world_points_from_depth=predictions.get("world_points_from_depth", None),
    )

    if frame_filter is None:
        frame_filter = "All"

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_"
        f"maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_"
        f"pred{prediction_mode.replace(' ', '_')}.glb",
    )

    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # ---- NEW: export all data (no images) + zip ----
    export_dir, zip_path = save_and_zip_results(target_dir, predictions, conf_thres_percent=float(conf_thres))

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success. Exported to:\n{export_dir}"

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True), export_dir, zip_path


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    return None

def update_log():
    return "Loading and Reconstructing..."

def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."
    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path, allow_pickle=True)
    predictions = {key: np.array(loaded[key]) if key in loaded else None for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_"
        f"maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_"
        f"pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Example videos
# -------------------------------------------------------------------------
great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
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
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }
    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a>
    </p>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=True, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="Upload Video", interactive=True)
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
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

            # NEW: Export outputs (folder + zip)
            export_folder_tb = gr.Textbox(label="Export Folder", interactive=False)
            export_zip_file  = gr.File(label="Download ZIP", interactive=False)

    # ---------------------- Examples section ----------------------
    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
        [room_video, "8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        [fern_video, "20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
    ]

    def example_pipeline(
        input_video,
        num_images_str,
        input_images,
        conf_thres_v,
        mask_black_bg_v,
        mask_white_bg_v,
        show_cam_v,
        mask_sky_v,
        prediction_mode_v,
        is_example_str,
    ):
        target_dir, image_paths = handle_uploads(input_video, input_images)
        frame_filter_v = "All"
        glbfile, log_msg, dropdown, export_dir, zip_path = gradio_demo(
            target_dir, conf_thres_v, frame_filter_v, mask_black_bg_v, mask_white_bg_v, show_cam_v, mask_sky_v, prediction_mode_v
        )
        # We only return the five outputs the Examples component expects:
        return glbfile, log_msg, target_dir, dropdown, image_paths

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
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
        outputs=[reconstruction_output, log_output, frame_filter, export_folder_tb, export_zip_file],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
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
