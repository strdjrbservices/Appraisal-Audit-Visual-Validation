from pipeline import process_image

# These values come from your existing Python + Regex extraction.
expected_label = "Garage"
image_path = "page12_image03.png"

result = process_image(
    image_path=image_path,
    expected_label=expected_label,
    output_json="page12_image03_result.json"
)

print(result["validation"]["status"])

# The original project can consume:
# result["paddleocr"]
# result["gemma"]
# result["validation"]
