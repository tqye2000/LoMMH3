###############################################################################
# Script to make an inpaint mask video for run_v2v.py by segmenting one object class in the source.
###############################################################################
import argparse
from pathlib import Path
import ltx_common as ltx
from run_v2v import read_video


def main():
    parser = argparse.ArgumentParser(description="Segment an object class in a video into a mask for run_v2v.py --mode inpaint.")
    parser.add_argument("--video", type=Path, required=True, help="Source video")
    parser.add_argument("--output", type=Path, default=Path("mask.mp4"))
    parser.add_argument("--label", default="person", help="COCO class to mask, e.g. person, cat, dog, car (default: %(default)s)")
    parser.add_argument("--score", type=float, default=0.5, help="Minimum detection confidence (default: %(default)s)")
    parser.add_argument("--grow", type=int, default=0, help="Dilate the mask by this many source pixels (default: %(default)s)")
    parser.add_argument("--box", action="store_true", help="Mask each detection's bounding box instead of its outline")
    parser.add_argument("--preview", type=Path, help="Also write the source with the mask tinted red")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    categories = weights.meta["categories"]
    if args.label not in categories or args.label in ("N/A", "__background__"):
        names = ", ".join(c for c in categories if c not in ("N/A", "__background__"))
        parser.error(f"--label must be one of: {names}")
    label = categories.index(args.label)

    try:
        # Resampled exactly as run_v2v.py does, so mask frame i matches source frame i there.
        frames = read_video(args.video, ltx.FPS)
    except Exception as error:
        parser.error(f"Cannot read source video: {error}")
    if not frames:
        parser.error("Source video has no frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().to(device)

    # H.264 needs even dimensions.
    width, height = frames[0].width // 2 * 2, frames[0].height // 2 * 2
    masks = torch.zeros(len(frames), height, width, dtype=torch.bool)
    with torch.inference_mode():
        for i, frame in enumerate(frames):
            image = torch.from_numpy(np.asarray(frame)).to(device).permute(2, 0, 1).float() / 255
            found = model([image])[0]
            pick = (found["labels"] == label) & (found["scores"] >= args.score)
            if args.box:
                mask = torch.zeros(image.shape[1:], dtype=torch.bool, device=device)
                for x0, y0, x1, y1 in found["boxes"][pick].round().int().tolist():
                    mask[y0 : y1 + 1, x0 : x1 + 1] = True
            else:
                mask = (found["masks"][pick, 0] > 0.5).any(0)
            if args.grow:
                mask = F.max_pool2d(mask[None].float(), 2 * args.grow + 1, 1, args.grow)[0] > 0
            masks[i] = mask[:height, :width].cpu()
            print(f"\rSegmented {i + 1}/{len(frames)} frames", end="", flush=True)
    print()

    covered = masks.flatten(1).float().mean(1)
    missing = [i for i, c in enumerate(covered.tolist()) if c == 0]
    print(f"Mask covers {covered.mean():.1%} of the frame on average, {covered.min():.1%}-{covered.max():.1%} per frame")
    if missing:
        print(f"No '{args.label}' found in {len(missing)} of {len(frames)} frames: {missing[:20]}{' ...' if len(missing) > 20 else ''}")

    ltx.save(args.output, (masks.to(torch.uint8) * 255)[..., None].repeat(1, 1, 1, 3), None, None)
    if args.preview:
        preview = torch.from_numpy(np.stack([np.asarray(f)[:height, :width] for f in frames]))
        preview[masks] = preview[masks] // 2 + torch.tensor([127, 0, 0], dtype=torch.uint8)
        ltx.save(args.preview, preview, None, None)


if __name__ == "__main__":
    main()
