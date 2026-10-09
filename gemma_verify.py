import os
import json
import base64
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

def _verify_with_ollama(image_path: str, prompt: str, model_name: str) -> dict:
    import io
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        max_size = 512
        if max(rgb_img.size) > max_size:
            rgb_img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=80)
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
            "num_predict": 250
        }
    }

    resp = requests.post(endpoint, json=payload, timeout=180)
    resp.raise_for_status()
    res_data = resp.json()
    content = res_data.get("message", {}).get("content", "")
    
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])
    return json.loads(content)

def verify_image(image_path: str, expected_label: str) -> dict:
    """
    Verify whether the image accurately matches the expected appraisal photo label
    using local Ollama (e.g. gemma3:4b, llama3.2-vision) or local HuggingFace Transformers.
    """
    prompt = f"""You are an appraisal image verification system.

Expected label: "{expected_label}"

Analyze the image and determine whether the expected label correctly describes the object/area shown.

Return ONLY valid JSON matching this schema:
{{
  "expected_label": "{expected_label}",
  "label_present": true,
  "label_correct": true,
  "observed_object": "string describing what is actually visible",
  "confidence": 0.95,
  "reason": "concise explanation"
}}

Rules:
- label_present: whether the expected feature or label context is present/relevant.
- label_correct: true if the image depicts what the expected label claims (e.g. kitchen, bedroom, front exterior, water heater, breaker panel, etc.).
- confidence: float between 0.0 and 1.0.
- Return raw JSON only, no markdown backticks.
"""

    # Check if Ollama is accessible
    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

    try:
        tag_resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3.0)
        if tag_resp.status_code == 200:
            available_models = [m.get("name", "") for m in tag_resp.json().get("models", [])]
            selected_model = ollama_model
            if not any(ollama_model in m for m in available_models):
                for m in available_models:
                    if any(v in m.lower() for v in ["gemma3:4b", "gemma3:12b", "llama3.2-vision", "llava", "minicpm-v", "qwen2.5-vl"]):
                        selected_model = m
                        break

            return _verify_with_ollama(image_path, prompt, selected_model)
    except Exception as e:
        if "Read timed out" in str(e) or "ConnectionRefused" not in str(e):
            # If Ollama is running but failed or errored, re-raise directly to show helpful Ollama error
            raise RuntimeError(f"Ollama vision verification error: {e}")
        pass

    # Fallback to local Hugging Face transformers
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

    start = decoded.find("{")
    end = decoded.rfind("}") + 1

    if start < 0 or end <= start:
        decoded_full = processor.decode(output[0], skip_special_tokens=True)
        start = decoded_full.find("{")
        end = decoded_full.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("Local model did not return valid JSON: " + decoded)
        decoded = decoded_full

    return json.loads(decoded[start:end])



