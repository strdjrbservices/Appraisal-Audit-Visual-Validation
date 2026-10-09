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
            try:
                from paddleocr import PaddleOCR
                cls._shared_ocr = PaddleOCR(lang=lang, show_log=False)
                cls._backend = "paddleocr"
            except Exception as e:
                cls._shared_ocr = None
                cls._backend = "none"
        return cls._shared_ocr, cls._backend

    def extract(self, image_path: str) -> dict:
        ocr, backend = self._get_ocr(self.lang)
        if not ocr or backend == "none":
            return {
                "image_path": image_path,
                "text": [],
                "boxes": [],
                "confidence": [],
                "raw_result": None
            }

        try:
            if backend == "rapidocr":
                raw_out = ocr(image_path)
                # RapidOCR can return a tuple (result, elapse) or result object
                result = raw_out[0] if isinstance(raw_out, tuple) else raw_out
                
                texts = []
                boxes = []
                scores = []

                if result:
                    if hasattr(result, "txts") and result.txts:
                        texts = list(result.txts)
                        if hasattr(result, "boxes") and result.boxes is not None:
                            boxes = [b.tolist() if hasattr(b, "tolist") else b for b in result.boxes]
                        if hasattr(result, "scores") and result.scores:
                            scores = [float(s) for s in result.scores]
                    elif isinstance(result, list):
                        for item in result:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                boxes.append(item[0].tolist() if hasattr(item[0], "tolist") else item[0])
                                if isinstance(item[1], (tuple, list)):
                                    texts.append(str(item[1][0]))
                                    if len(item[1]) > 1:
                                        scores.append(float(item[1][1]))
                                else:
                                    texts.append(str(item[1]))
                                    if len(item) >= 3:
                                        scores.append(float(item[2]))
                
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
                result = ocr.ocr(image_path, cls=True)
                texts = []
                boxes = []
                scores = []
                if result and isinstance(result, list) and len(result) > 0 and result[0]:
                    for line in result[0]:
                        if len(line) >= 2:
                            boxes.append(line[0])
                            texts.append(str(line[1][0]))
                            scores.append(float(line[1][1]))
                return {
                    "image_path": image_path,
                    "text": texts,
                    "boxes": boxes,
                    "confidence": scores,
                    "raw_result": result
                }
        except Exception as e:
            return {
                "image_path": image_path,
                "text": [],
                "boxes": [],
                "confidence": [],
                "raw_result": str(e)
            }

    @staticmethod
    def save(data: dict, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
