import json
import os
import numpy as np
from PIL import Image

class PaddleOCRExtractor:
    _shared_ocr = None
    _backend = None

    def __init__(self, lang="en"):
        self.lang = lang

    @classmethod
    def _get_ocr(cls, lang="en"):
        if cls._shared_ocr is not None:
            return cls._shared_ocr, cls._backend
        try:
            from rapidocr import RapidOCR
            cls._shared_ocr = RapidOCR()
            cls._backend = "rapidocr"
        except Exception:
            from paddleocr import PaddleOCR
            cls._shared_ocr = PaddleOCR(lang=lang)
            cls._backend = "paddleocr"
        return cls._shared_ocr, cls._backend

    def extract(self, image_path: str) -> dict:
        ocr, backend = self._get_ocr(self.lang)
        if backend == "rapidocr":
            # RapidOCR accepts path, PIL Image, or numpy array
            result = ocr(image_path)
            # result has .txts, .boxes, .scores
            texts = list(result.txts) if result and result.txts else []
            boxes = [b.tolist() if hasattr(b, "tolist") else b for b in result.boxes] if result and result.boxes is not None else []
            scores = [float(s) for s in result.scores] if result and result.scores else []
            return {
                "image_path": image_path,
                "text": texts,
                "boxes": boxes,
                "confidence": scores,
                "raw_result": {
                    "txts": texts,
                    "scores": scores
                }
            }
        else:
            result = ocr.predict(image_path)
            return {
                "image_path": image_path,
                "raw_result": result
            }

    @staticmethod
    def save(data: dict, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

