from __future__ import annotations
import os
import io
import hashlib
from typing import Iterable, Tuple, Dict

from .logging_setup import logger

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (pymupdf) is required. Please install it.") from e

try:
    import pytesseract
    from PIL import Image
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False
    logger.warning("pytesseract/Pillow not found; OCR will be disabled.")


def _clean_text(s: str) -> str:
    # Normalize whitespace and drop extremely short boilerplate lines
    lines = [ln.strip() for ln in s.replace('\r', '\n').split('\n')]
    lines = [ln for ln in lines if ln]
    return '\n'.join(lines)


def iter_docs(path: str) -> Iterable[Tuple[str, str, Dict]]:
    """
    Yield (unit_id, text, metadata) for each logical unit:
    - For PDFs: each page becomes a unit
    - For images: entire image is a unit (OCR if available)
    - For .txt/.md: whole file is one unit
    """
    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in {".pdf"}:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text") or ""
                text = _clean_text(text)
                if not text and _HAS_TESS:
                    # Light OCR only if page has no text
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        text = _clean_text(pytesseract.image_to_string(img))
                    except Exception as e:
                        logger.debug(f"OCR failed on {path} page {i+1}: {e}")
                meta = {
                    "source": path,
                    "type": "pdf",
                    "page": i + 1,
                    "pages": len(doc),
                }
                if text:
                    yield (_unit_id(path, i), text, meta)

    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        if not _HAS_TESS:
            logger.info(f"Skipping image without OCR: {path}")
            return
        try:
            img = Image.open(path)
            text = _clean_text(pytesseract.image_to_string(img))
        except Exception as e:
            logger.debug(f"OCR failed on image {path}: {e}")
            return
        if text:
            meta = {"source": path, "type": "image"}
            yield (_unit_id(path, 0), text, meta)

    elif ext in {".txt", ".md"}:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = _clean_text(f.read())
        except Exception as e:
            logger.debug(f"Read failed {path}: {e}")
            return
        if text:
            meta = {"source": path, "type": "text"}
            yield (_unit_id(path, 0), text, meta)

    else:
        # Unsupported types are silently ignored
        return


def _unit_id(path: str, idx: int) -> str:
    h = hashlib.sha1(f"{path}:{idx}".encode("utf-8")).hexdigest()[:12]
    return f"{os.path.basename(path)}::{idx+1}::{h}"