# Appraisal Image Validation

Architecture:

PDF
  -> existing Python + Regex
  -> expected structured values

Image region
  -> PaddleOCR
  -> OCR evidence JSON

Image region
  -> Gemma Vision
  -> image/label verification JSON

OCR + Gemma + expected values
  -> validation engine
  -> PASS / FAIL / REVIEW

Keep PaddleOCR-specific parsing in normalize_paddleocr().
Keep Gemma-specific inference in gemma_verify.py.
The original application should consume only the stable JSON contract.
"# Appraisal-Audit-Visual-Validation" 
