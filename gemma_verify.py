import os
import json
import time
import base64
import re
import requests
from PIL import Image

_processor = None
_model = None

def _get_local_transformers_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    MODEL = os.environ.get("GEMMA_MODEL_NAME", "google/gemma-3-4b-it")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or True
    
    _processor = AutoProcessor.from_pretrained(MODEL, token=token)
    
    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    _model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=token
    )
    if not torch.cuda.is_available():
        _model = _model.to("cpu")
        
    return _processor, _model

def _clean_and_parse_json(content: str) -> dict:
    if not content:
        raise ValueError("Empty response content from vision model.")
    
    # Strip markdown code blocks
    content_cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.MULTILINE)
    content_cleaned = re.sub(r"```\s*$", "", content_cleaned.strip(), flags=re.MULTILINE)
    
    start = content_cleaned.find("{")
    end = content_cleaned.rfind("}") + 1
    
    if start >= 0 and end > start:
        json_str = content_cleaned[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try relaxing trailing commas
            relaxed = re.sub(r",\s*([\]}])", r"\1", json_str)
            return json.loads(relaxed)
            
    return json.loads(content_cleaned)

def _verify_with_ollama(image_path: str, prompt: str, model_name: str) -> dict:
    import io
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        max_size = 768
        if max(rgb_img.size) > max_size:
            rgb_img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    endpoint = f"{ollama_url.rstrip('/')}/api/chat"
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64]
            }
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300
        }
    }

    max_retries = 2
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(endpoint, json=payload, timeout=240)
            resp.raise_for_status()
            res_data = resp.json()
            content = res_data.get("message", {}).get("content", "")
            return _clean_and_parse_json(content)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
            else:
                raise last_err

def verify_image(image_path: str, expected_label: str) -> dict:
    """
    Verify whether the image accurately matches the expected appraisal photo label
    using local Ollama (e.g. gemma3:4b, llama3.2-vision) or local HuggingFace Transformers.
    """
    prompt = f"""You are an appraisal image verification system.

Expected appraisal photo label: "{expected_label}"

Analyze the provided appraisal photograph and determine whether it accurately depicts what the expected label describes.

Return ONLY valid JSON matching this schema:
{{
  "expected_label": "{expected_label}",
  "label_present": true,
  "label_correct": true,
  "observed_object": "concise description of what is visible in the photo",
  "confidence": 0.95,
  "reason": "concise explanation of whether the photo matches the expected label"
}}

Rules:
- label_present: boolean (true if the expected feature or label context is present/relevant).
- label_correct: boolean (true if the image depicts what the label claims, e.g. front exterior, rear exterior, street scene, kitchen, bathroom, bedroom, living room, laundry/utility, water heater, electrical panel, smoke detector, building sketch, or location map).
- confidence: float between 0.0 and 1.0.
- Return raw JSON only, no markdown backticks.
"""

    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

    try:
        tag_resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3.0)
        if tag_resp.status_code == 200:
            available_models = [m.get("name", "") for m in tag_resp.json().get("models", [])]
            selected_model = ollama_model
            if not any(ollama_model in m for m in available_models):
                for m in available_models:
                    if any(v in m.lower() for v in ["gemma3:4b", "gemma3:12b", "gemma3", "llama3.2-vision", "llava", "minicpm-v", "qwen2.5-vl"]):
                        selected_model = m
                        break

            raw_res = _verify_with_ollama(image_path, prompt, selected_model)
            return _normalize_verification_schema(raw_res, expected_label)
    except Exception as e:
        if "ConnectionRefused" not in str(e) and "Failed to establish a new connection" not in str(e):
            # If Ollama is running but failed after retries, return structured error rather than breaking whole pipeline
            return {
                "expected_label": expected_label,
                "label_present": False,
                "label_correct": False,
                "observed_object": f"Vision verification unavailable: {e}",
                "confidence": 0.0,
                "reason": f"Ollama vision processing error: {e}"
            }

    # Fallback to local Hugging Face transformers if Ollama not available
    try:
        import torch
        processor, model = _get_local_transformers_model()
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        try:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
        except Exception:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300)

        input_len = inputs.get("input_ids", torch.empty(1, 0)).shape[1]
        gen_tokens = output[0][input_len:] if input_len > 0 else output[0]
        decoded = processor.decode(gen_tokens, skip_special_tokens=True)

        raw_res = _clean_and_parse_json(decoded)
        return _normalize_verification_schema(raw_res, expected_label)
    except Exception as e:
        return {
            "expected_label": expected_label,
            "label_present": False,
            "label_correct": False,
            "observed_object": f"Transformers error: {e}",
            "confidence": 0.0,
            "reason": str(e)
        }

def _normalize_verification_schema(raw_dict: dict, fallback_label: str) -> dict:
    if not isinstance(raw_dict, dict):
        return {
            "expected_label": fallback_label,
            "label_present": False,
            "label_correct": False,
            "observed_object": str(raw_dict),
            "confidence": 0.0,
            "reason": "Invalid response format from vision model."
        }
    
    # Safe boolean normalization
    raw_present = raw_dict.get("label_present", True)
    if isinstance(raw_present, str):
        label_present = raw_present.strip().lower() in ("true", "1", "yes", "y", "pass")
    else:
        label_present = bool(raw_present)

    raw_correct = raw_dict.get("label_correct", False)
    if isinstance(raw_correct, str):
        label_correct = raw_correct.strip().lower() in ("true", "1", "yes", "y", "pass")
    else:
        label_correct = bool(raw_correct)

    raw_conf = raw_dict.get("confidence", 0.0)
    try:
        if isinstance(raw_conf, str):
            raw_conf = raw_conf.replace("%", "").strip()
            confidence = float(raw_conf)
            if confidence > 1.0:
                confidence = confidence / 100.0
        else:
            confidence = float(raw_conf) if raw_conf is not None else 0.0
    except (ValueError, TypeError):
        confidence = 0.0

    return {
        "expected_label": raw_dict.get("expected_label") or fallback_label,
        "label_present": label_present,
        "label_correct": label_correct,
        "observed_object": str(raw_dict.get("observed_object") or "").strip(),
        "confidence": round(min(1.0, max(0.0, confidence)), 2),
        "reason": str(raw_dict.get("reason") or "").strip()
    }
