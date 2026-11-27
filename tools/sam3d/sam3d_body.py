"""
SAM 3D Body (3DB): single-image full-body 3D human mesh recovery.

Models live under D:\\hf_models (WSL: /mnt/d/hf_models, override with SAM3D_HF_ROOT):
  sam-3d-body-code/        git clone of facebookresearch/sam-3d-body
  sam-3d-body-<variant>/   gated checkpoint (model.ckpt, model_config.yaml, assets/mhr_model.pt)
  vitdet/                  ViTDet human detector (model_final_f05665.pkl)
  moge-2-vitl-normal/      MoGe-2 FOV estimator (model.pt)

Usage:
    python tools/sam3d/sam3d_body.py download [--variant dinov3|vith]      # Windows .venv (HF Hub reachable)
    python tools/sam3d/sam3d_body.py image --images images --out outputs/sam3d [--variant dinov3] [--no-detector] [--no-fov]
    python tools/sam3d/sam3d_body.py video --input clip.mp4 --out outputs/sam3d/clip_white.mp4 [--background black|white|video]
"""

import argparse
import os
import sys
from pathlib import Path

HF_ROOT = Path(os.environ.get("SAM3D_HF_ROOT", r"D:\hf_models" if os.name == "nt" else "/mnt/d/hf_models"))
CODE_DIR = HF_ROOT / "sam-3d-body-code"
VARIANTS = {"dinov3": "facebook/sam-3d-body-dinov3", "vith": "facebook/sam-3d-body-vith"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
WHITE_MODEL = (0.85, 0.85, 0.85)


def ckpt_dir(variant: str) -> Path:
    return HF_ROOT / f"sam-3d-body-{variant}"


def download(variant: str) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    repo = VARIANTS[variant]
    try:
        path = snapshot_download(repo, local_dir=ckpt_dir(variant))
    except GatedRepoError:
        sys.exit(
            f"No access to gated repo {repo}. Request access at https://huggingface.co/{repo}, "
            "wait for approval, then log in (`hf auth login`) and rerun."
        )
    print(f"Downloaded {repo} -> {path}")


def list_images(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def build_estimator(args: argparse.Namespace):
    sys.path.insert(0, str(CODE_DIR))
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    d = ckpt_dir(args.variant)
    ckpt, mhr = d / "model.ckpt", d / "assets" / "mhr_model.pt"
    if not ckpt.exists():
        sys.exit(f"Checkpoint not found: {ckpt}. Run `python tools/sam3d/sam3d_body.py download --variant {args.variant}` first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_sam_3d_body(str(ckpt), device=device, mhr_path=str(mhr))

    detector = fov = None
    if not args.no_detector:
        from tools.build_detector import HumanDetector

        detector = HumanDetector(name="vitdet", device=device, path=str(HF_ROOT / "vitdet"))
    if not args.no_fov:
        from tools.build_fov_estimator import FOVEstimator

        fov = FOVEstimator(name="moge2", device=device, path=str(HF_ROOT / "moge-2-vitl-normal" / "model.pt"))

    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=detector,
        human_segmentor=None,
        fov_estimator=fov,
    )


def image(args: argparse.Namespace) -> None:
    estimator = build_estimator(args)
    import cv2
    import numpy as np
    from sam_3d_body.visualization.renderer import Renderer
    from tools.vis_utils import visualize_sample_together

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for img_path in list_images(Path(args.images)):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"skip (unreadable): {img_path}")
            continue
        outputs = estimator.process_one_image(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), bbox_thr=args.bbox_thresh
        )
        stem = img_path.stem
        print(f"{img_path.name}: {len(outputs)} person(s)")
        if not outputs:
            continue

        cv2.imwrite(str(out / f"{stem}_vis.jpg"), visualize_sample_together(img_bgr, outputs, estimator.faces).astype(np.uint8))
        for i, person in enumerate(outputs):
            renderer = Renderer(focal_length=person["focal_length"], faces=estimator.faces)
            mesh = renderer.vertices_to_trimesh(person["pred_vertices"], person["pred_cam_t"], LIGHT_BLUE)
            mesh.export(out / f"{stem}_mesh_{i:02d}.ply")
            arrays = {k: np.asarray(v) for k, v in person.items() if isinstance(v, (np.ndarray, float, int))}
            np.savez(out / f"{stem}_params_{i:02d}.npz", **arrays)


class WhiteModelRenderer:
    """Renders every person of a frame into one scene so occlusions between people are correct."""

    def __init__(self, faces, width: int, height: int):
        import pyrender

        self.faces = faces
        self.width, self.height = width, height
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, roughnessFactor=0.8, alphaMode="OPAQUE", baseColorFactor=(*WHITE_MODEL, 1.0)
        )

    def __call__(self, outputs, background):
        import numpy as np
        import pyrender
        import trimesh
        from sam_3d_body.visualization.renderer import create_raymond_lights

        if not outputs:
            return background
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        # Same camera convention as Renderer.render_rgba_multiple: camera at origin, mesh flipped about x.
        flip = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        for person in outputs:
            mesh = trimesh.Trimesh(person["pred_vertices"] + person["pred_cam_t"], self.faces, process=False)
            mesh.apply_transform(flip)
            scene.add(pyrender.Mesh.from_trimesh(mesh, material=self.material))
        focal = float(outputs[0]["focal_length"])
        camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=self.width / 2, cy=self.height / 2, zfar=1e12)
        scene.add(camera, pose=np.eye(4))
        for node in create_raymond_lights():
            scene.add_node(node)
        rgba, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        return (rgba[..., :3] * alpha + background * (1.0 - alpha)).astype(np.uint8)


def video(args: argparse.Namespace) -> None:
    import contextlib
    import io

    estimator = build_estimator(args)
    import av
    import numpy as np
    from tqdm import tqdm

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with av.open(args.input) as src, av.open(str(dst), "w") as out:
        vin = src.streams.video[0]
        ain = src.streams.audio[0] if src.streams.audio and not args.no_audio else None
        # libx264 + yuv420p needs even dimensions.
        w, h = vin.codec_context.width // 2 * 2, vin.codec_context.height // 2 * 2
        vout = out.add_stream("libx264", rate=vin.average_rate or 30)
        vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
        vout.options = {"crf": "18"}

        aout, acopy = None, False
        if ain is not None:
            # AAC/MP3 can be stream-copied into mp4; anything else (pcm, vorbis, ...) is re-encoded to AAC.
            acopy = ain.codec_context.name in {"aac", "mp3"}
            if acopy:
                add_tpl = getattr(out, "add_stream_from_template", None)
                aout = add_tpl(ain) if add_tpl else out.add_stream(template=ain)
            else:
                aout = out.add_stream("aac", rate=ain.codec_context.sample_rate)

        render = WhiteModelRenderer(estimator.faces, w, h)
        bg_fill = {"black": 0, "white": 255}.get(args.background)
        cam_int = None

        total = vin.frames or None
        stop = None
        if args.max_frames:
            total = min(total or args.max_frames, args.max_frames)
            stop = args.max_frames / float(vin.average_rate or 30)

        i = 0
        bar = tqdm(total=total, unit="frame")
        for packet in src.demux(*[s for s in (vin, ain) if s is not None]):
            if packet.stream is ain:
                if stop is not None and packet.pts is not None and packet.pts * packet.time_base >= stop:
                    continue
                if acopy:
                    if packet.dts is None:
                        continue
                    packet.stream = aout
                    out.mux(packet)
                else:
                    for aframe in packet.decode():
                        out.mux(aout.encode(aframe))
                continue

            if args.max_frames and i >= args.max_frames:
                continue
            for frame in packet.decode():
                if args.max_frames and i >= args.max_frames:
                    break
                rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24")[:h, :w])
                if cam_int is None and estimator.fov_estimator is not None and not args.per_frame_fov:
                    # Fixed intrinsics from the first frame avoid focal-length jitter between frames.
                    cam_int = estimator.fov_estimator.get_cam_intrinsics(rgb)
                with contextlib.redirect_stdout(io.StringIO()):
                    outputs = estimator.process_one_image(rgb, cam_int=cam_int, bbox_thr=args.bbox_thresh)
                background = rgb if bg_fill is None else np.full_like(rgb, bg_fill)
                img = render(outputs, background)
                out.mux(vout.encode(av.VideoFrame.from_ndarray(img, format="rgb24")))
                i += 1
                bar.update()
        bar.close()
        out.mux(vout.encode())
        if aout is not None and not acopy:
            out.mux(aout.encode())
    print(f"Saved {dst}" + (" (with audio)" if aout is not None else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download the gated 3DB checkpoint to HF_ROOT")
    p_dl.add_argument("--variant", choices=VARIANTS, default="dinov3")

    # "run" is the old name, kept so existing commands still work.
    p_img = sub.add_parser("image", aliases=["run"], help="Reconstruct 3D human meshes from images")
    p_img.add_argument("--images", required=True, help="Image file or folder")
    p_img.add_argument("--out", default="outputs/sam3d")
    p_img.add_argument("--variant", choices=VARIANTS, default="dinov3")
    p_img.add_argument("--bbox-thresh", type=float, default=0.8)
    p_img.add_argument("--no-detector", action="store_true", help="Treat the whole image as one person")
    p_img.add_argument("--no-fov", action="store_true", help="Use default FOV instead of MoGe-2")

    p_vid = sub.add_parser("video", help="Turn a video into a white-model (untextured mesh) video")
    p_vid.add_argument("--input", required=True, help="Input video file")
    p_vid.add_argument("--out", required=True, help="Output .mp4")
    p_vid.add_argument("--variant", choices=VARIANTS, default="dinov3")
    p_vid.add_argument("--background", choices=["black", "white", "video"], default="black")
    p_vid.add_argument("--bbox-thresh", type=float, default=0.8)
    p_vid.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = all)")
    p_vid.add_argument("--per-frame-fov", action="store_true", help="Re-estimate FOV every frame (for zooming shots)")
    p_vid.add_argument("--no-audio", action="store_true", help="Drop the input audio track")
    p_vid.add_argument("--no-detector", action="store_true", help="Treat the whole frame as one person")
    p_vid.add_argument("--no-fov", action="store_true", help="Use default FOV instead of MoGe-2")

    args = parser.parse_args()
    if args.cmd == "download":
        download(args.variant)
    elif args.cmd == "video":
        video(args)
    else:
        image(args)


if __name__ == "__main__":
    main()
