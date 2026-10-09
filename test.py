import unittest
import os
import pymupdf
from address_validator import normalize_address, evaluate_address_match, extract_master_subject
from validation_engine import validate_image_evidence, _is_truthy, _parse_confidence
from gemma_verify import _normalize_verification_schema
from appraisal_report_generator import parse_photo_slots_from_page

class TestAppraisalValidationSuite(unittest.TestCase):

    def test_address_normalization(self):
        self.assertEqual(normalize_address("90 She Oak Dr SW"), "90 she oak dr sw")
        self.assertEqual(normalize_address("1806 W. 16th Street"), "1806 w 16th st")
        self.assertEqual(normalize_address("16 Firwood Drive, Apt 2B"), "16 firwood dr apt 2b")

    def test_validation_engine_truthiness_and_types(self):
        # Test Python string 'false' bug fix
        self.assertFalse(_is_truthy("false"))
        self.assertFalse(_is_truthy("False"))
        self.assertFalse(_is_truthy(False))
        self.assertTrue(_is_truthy("true"))
        self.assertTrue(_is_truthy("yes"))
        self.assertTrue(_is_truthy(True))

        # Test confidence parsing
        self.assertEqual(_parse_confidence("95%"), 0.95)
        self.assertEqual(_parse_confidence(0.98), 0.98)
        self.assertEqual(_parse_confidence(98), 0.98)
        self.assertEqual(_parse_confidence("invalid"), 0.0)

    def test_validation_evidence_logic(self):
        # Case 1: Full visual pass
        res = validate_image_evidence(
            expected_label="Subject Front",
            ocr_result={"text": []},
            gemma_result={"label_correct": True, "confidence": 0.95, "observed_object": "Exterior view of front facade"}
        )
        self.assertEqual(res["status"], "PASS")

        # Case 2: Semantic synonym match
        res2 = validate_image_evidence(
            expected_label="1/2 bath",
            ocr_result={"text": []},
            gemma_result={"label_correct": False, "confidence": 0.85, "observed_object": "A small bathroom with a toilet and sink"}
        )
        self.assertEqual(res2["status"], "PASS")

        # Case 3: Clear mismatch
        res3 = validate_image_evidence(
            expected_label="Kitchen",
            ocr_result={"text": []},
            gemma_result={"label_correct": False, "confidence": 0.90, "observed_object": "Residential street with parked cars"}
        )
        self.assertEqual(res3["status"], "FAIL")

    def test_gemma_schema_normalization(self):
        raw = {
            "expected_label": "Water Heater",
            "label_present": "true",
            "label_correct": "true",
            "observed_object": "Water heater tank in utility closet",
            "confidence": "92%",
            "reason": "Clear water heater unit"
        }
        norm = _normalize_verification_schema(raw, "Water Heater")
        self.assertTrue(norm["label_present"])
        self.assertTrue(norm["label_correct"])
        self.assertEqual(norm["confidence"], 0.92)

    def test_spatial_slot_parsing_if_pdf_exists(self):
        sample_pdf = r"C:\Users\Admin\Downloads\90 She Oak Dr SW.pdf"
        if os.path.exists(sample_pdf):
            doc = pymupdf.open(sample_pdf)
            # Page 11 (index 10) is the 14-photo addendum
            slots = parse_photo_slots_from_page(doc[10], 11)
            self.assertEqual(len(slots), 14)
            labels = [s["label"] for s in slots]
            # Verify correct captions are assigned to slots spatially
            self.assertIn("Front", labels[0])
            self.assertIn("Rear", labels[1])
            self.assertIn("Street", labels[2])
            self.assertIn("Garage", labels[3])
            self.assertIn("1/2 bath", labels[4])
            self.assertIn("Family Room", labels[5])
            self.assertIn("Kitchen", labels[6])

if __name__ == "__main__":
    unittest.main()
