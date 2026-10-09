import json
from paddleocr_module import PaddleOCRExtractor
from gemma_verify import verify_image
from validation_engine import validate_image_evidence

def normalize_paddleocr(raw_result):
    """
    Convert OCR response into a stable format for appraisal validation.
    """
    if isinstance(raw_result, dict) and "text" in raw_result:
        return {
            "text": raw_result.get("text", []),
            "boxes": raw_result.get("boxes", []),
            "confidence": raw_result.get("confidence", []),
            "raw": raw_result.get("raw_result")
        }

    texts = []
    boxes = []
    scores = []
    raw = raw_result.get("raw_result", raw_result) if isinstance(raw_result, dict) else raw_result
    
    if isinstance(raw, list):
        for line in raw:
            if isinstance(line, list) and len(line) >= 2:
                boxes.append(line[0])
                if isinstance(line[1], (tuple, list)):
                    texts.append(str(line[1][0]))
                    if len(line[1]) > 1:
                        scores.append(float(line[1][1]))
            elif isinstance(line, dict):
                texts.append(str(line.get("text", "")))
                scores.append(float(line.get("score", 0.0)))
    
    return {
        "text": texts,
        "boxes": boxes,
        "confidence": scores,
        "raw": raw
    }


def process_image(image_path, expected_label, output_json):
    # 1. OCR evidence
    ocr_raw = PaddleOCRExtractor(lang="en").extract(image_path)
    ocr = normalize_paddleocr(ocr_raw)

    # 2. Visual verification
    gemma = verify_image(image_path, expected_label)

    # 3. Appraisal-specific validation
    validation = validate_image_evidence(
        expected_label,
        ocr,
        gemma
    )

    # 4. Common JSON contract
    result = {
        "image_path": image_path,
        "expected_label": expected_label,
        "paddleocr": ocr,
        "gemma": gemma,
        "validation": validation
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result
