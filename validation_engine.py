import re

def _is_truthy(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val > 0
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "y", "pass", "match", "correct")
    return False

def _parse_confidence(val):
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        c = float(val)
        return min(1.0, max(0.0, c if c <= 1.0 else c / 100.0))
    if isinstance(val, str):
        cleaned = val.replace("%", "").strip()
        try:
            c = float(cleaned)
            return min(1.0, max(0.0, c if c <= 1.0 else c / 100.0))
        except (ValueError, TypeError):
            return 0.0
    return 0.0

def validate_image_evidence(expected_label: str, ocr_result: dict, gemma_result: dict) -> dict:
    """
    Appraisal Image Evidence Validation Engine:
    Combines Vision Model (Gemma) verification with PaddleOCR extracted tokens
    and domain-specific appraisal rules.
    """
    expected = (expected_label or "").strip()
    exp_lower = expected.lower()

    # OCR Tokens
    ocr_texts = [str(x).strip() for x in (ocr_result or {}).get("text", []) if str(x).strip()]
    ocr_combined = " ".join(ocr_texts).lower()

    # Gemma visual attributes
    gemma_match = _is_truthy(gemma_result.get("label_correct", False))
    confidence = _parse_confidence(gemma_result.get("confidence", 0.0))
    observed = (gemma_result.get("observed_object") or "").strip()
    obs_lower = observed.lower()

    # 1. Direct OCR Match check
    # Check if key words from label appear in OCR text (e.g. street name, comp number, room name)
    ocr_match = False
    if expected:
        # Extract meaningful alphanumeric tokens from expected label (ignoring generic words)
        tokens = [t for t in re.findall(r"[A-Za-z0-9]+", exp_lower) if len(t) > 2 and t not in ("photo", "slot", "view", "page", "the", "and")]
        if tokens:
            ocr_match = any(t in ocr_combined for t in tokens)

    # 2. Semantic synonym mapping between expected label and observed description
    semantic_synonym_match = False
    
    # Rules dictionary for appraisal photo types:
    appraisal_domains = {
        "bathroom": ["bath", "toilet", "vanity", "shower", "tub", "sink", "restroom", "powder room", "1/2 bath", "half bath", "full bath"],
        "kitchen": ["kitchen", "cabinets", "refrigerator", "stove", "oven", "range", "countertop", "sink", "dishwasher", "microwave"],
        "bedroom": ["bedroom", "bed", "closet", "sleeping", "room with window", "blank room", "carpeted room"],
        "living": ["living room", "family room", "great room", "den", "seating", "couch", "sofa", "fireplace", "living"],
        "front": ["front", "front exterior", "facade", "front entrance", "driveway", "lawn", "front elevation", "townhouse", "house"],
        "rear": ["rear", "back exterior", "backyard", "patio", "deck", "rear elevation", "back wall", "gutters"],
        "street": ["street", "road", "neighborhood", "cul-de-sac", "residential street", "paved", "curb", "cars", "vehicles"],
        "garage": ["garage", "carport", "garage door", "concrete floor", "drywall", "parking"],
        "utility": ["utility", "laundry", "washer", "dryer", "hookup", "mechanical", "furnace", "hvac"],
        "water heater": ["water heater", "tank", "boiler", "hot water"],
        "panel": ["breaker", "panel", "electrical panel", "fuse box", "circuit breaker"],
        "smoke detector": ["smoke detector", "co detector", "carbon monoxide", "detector", "alarm", "sensor"],
        "sketch": ["sketch", "floor plan", "layout", "diagram", "dimensions", "drawing", "gross living area", "gla"],
        "map": ["map", "aerial", "satellite", "comparable location", "streets", "parcel", "plat", "markers", "location"],
        "comp": ["comparable", "house", "townhouse", "residential", "exterior", "two-story", "single family", "home"]
    }

    for domain_key, syns in appraisal_domains.items():
        if any(s in exp_lower for s in syns):
            if any(s in obs_lower for s in syns):
                semantic_synonym_match = True
                break

    # 3. Final Decision Logic
    if (gemma_match and confidence >= 0.70) or (semantic_synonym_match and confidence >= 0.65):
        status = "PASS"
        reason = "Visual verification confirmed: Image accurately depicts the expected appraisal feature."
    elif gemma_match and confidence >= 0.50:
        status = "PASS"
        reason = "Visual verification matches expected label with moderate confidence."
    elif semantic_synonym_match:
        status = "PASS"
        reason = f"Observed visual elements ({observed[:50]}...) align with expected label '{expected}'."
    elif ocr_match:
        status = "REVIEW"
        reason = "OCR text overlay aligns with label, but visual verification requires secondary review."
    elif confidence == 0.0 and "error" in (gemma_result.get("reason", "") + obs_lower):
        status = "REVIEW"
        reason = f"Verification inconclusive due to processing notice: {gemma_result.get('reason') or observed}"
    else:
        status = "FAIL"
        reason = f"Image evidence does not sufficiently support '{expected}' (Observed: {observed or 'Unrecognized content'})."

    return {
        "expected_label": expected,
        "ocr_label_match": ocr_match,
        "gemma_label_correct": gemma_match or semantic_synonym_match,
        "gemma_confidence": confidence,
        "status": status,
        "reason": reason
    }
