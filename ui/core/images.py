"""Validate uploaded images and store them immutably with thumbnails."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from . import constants as C
from .models import ImageAsset, new_id, utc_now


class ImageRejected(ValueError):
    pass


def ingest_image(project_dir: Path, display_name: str, data: bytes) -> ImageAsset:
    """Validate ``data`` as an image, store it under ``assets/``, and return its record."""
    if len(data) > C.MAX_UPLOAD_BYTES:
        raise ImageRejected(f"{display_name}: larger than {C.MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        image = Image.open(io.BytesIO(data))
        image_format = image.format or ""
        width, height = image.size
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageRejected(f"{display_name}: not a readable image ({exc})") from exc
    if image_format not in C.ALLOWED_IMAGE_FORMATS:
        raise ImageRejected(f"{display_name}: unsupported format {image_format or 'unknown'}")
    if width * height > C.MAX_IMAGE_PIXELS:
        raise ImageRejected(f"{display_name}: image is too large ({width}x{height})")

    asset_id = new_id("img")
    ext = C.ALLOWED_IMAGE_FORMATS[image_format]
    assets = project_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / f"{asset_id}{ext}").write_bytes(data)

    thumb = ImageOps.exif_transpose(image).convert("RGB")
    thumb.thumbnail(C.THUMB_SIZE)
    thumb.save(assets / f"{asset_id}_thumb.jpg", format="JPEG", quality=85)
    image.close()

    safe_name = Path(display_name).name[:120] or f"{asset_id}{ext}"
    return ImageAsset(
        id=asset_id,
        name=safe_name,
        path=f"assets/{asset_id}{ext}",
        thumb=f"assets/{asset_id}_thumb.jpg",
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        added_utc=utc_now(),
    )
