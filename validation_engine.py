def validate_image_evidence(expected_label, ocr_result, gemma_result):
    ocr_text = " ".join(
        str(x) for x in ocr_result.get("text", [])
    ).lower()

    expected = expected_label.lower().strip()
    ocr_match = expected in ocr_text if expected else False

    gemma_match = bool(gemma_result.get("label_correct", False))
    confidence = float(gemma_result.get("confidence", 0))

    if gemma_match and confidence >= 0.80:
        status = "PASS"
        reason = "Gemma confirms that the image matches the expected label."
    elif ocr_match:
        status = "REVIEW"
        reason = "OCR supports the label, but visual verification needs review."
    else:
        status = "FAIL"
        reason = "Image evidence does not sufficiently support the expected label."

    return {
        "expected_label": expected_label,
        "ocr_label_match": ocr_match,
        "gemma_label_correct": gemma_match,
        "gemma_confidence": confidence,
        "status": status,
        "reason": reason
    }
