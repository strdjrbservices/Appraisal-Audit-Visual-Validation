import os
import sys
import json
import pymupdf
from PIL import Image
from pipeline import process_image
from validation_engine import validate_image_evidence

def extract_and_validate_pdf(pdf_path: str, output_base_dir: str = "validation_output"):
    os.makedirs(output_base_dir, exist_ok=True)
    images_dir = os.path.join(output_base_dir, "images")
    results_dir = os.path.join(output_base_dir, "json_results")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    print(f"Loaded PDF: {pdf_path} ({len(doc)} pages)")

    # Define the photo extraction specifications based on Fannie Mae / TOTAL appraisal standard layouts
    # We define pages and their labeled photo slots
    # Format: (page_number_1_indexed, slot_index, expected_label, category, rect_tuple_pct or xref)
    
    # Let's inspect pages dynamically
    jobs = []

    # Page 10: Subject Photos (3 vertical slots)
    # Slots: Front, Rear, Street
    jobs.extend([
        {"page": 10, "slot": "Subject_Front", "label": "Subject Front (1806 W 16th St)", "crop_box": (30, 80, 540, 360)},
        {"page": 10, "slot": "Subject_Rear", "label": "Subject Rear (1806 W 16th St)", "crop_box": (30, 370, 540, 650)},
        {"page": 10, "slot": "Subject_Street", "label": "Subject Street Scene (1806 W 16th St)", "crop_box": (30, 660, 540, 950)},
    ])

    # Page 11: Comparable Photos 1-3 (3 vertical slots)
    jobs.extend([
        {"page": 11, "slot": "Comparable_1", "label": "Comparable 1 (1321 W 23rd St)", "crop_box": (30, 80, 540, 360)},
        {"page": 11, "slot": "Comparable_2", "label": "Comparable 2 (1700 Schaer St)", "crop_box": (30, 370, 540, 650)},
        {"page": 11, "slot": "Comparable_3", "label": "Comparable 3 (1223 W 10th St)", "crop_box": (30, 660, 540, 950)},
    ])

    # Page 12: Comparable Photo 4
    jobs.extend([
        {"page": 12, "slot": "Comparable_4", "label": "Comparable 4 (2207 W Long 17th St)", "crop_box": (30, 80, 540, 360)},
    ])

    # Page 13: Photograph Addendum (5 rows x 3 cols = 15 photos)
    # Row 1 (y: ~60 to ~200): Front/Left Exterior, Front/Right Exterior, Rear/Left Exterior
    # Row 2 (y: ~200 to ~340): Rear/Right Exterior, Living Room, Additional View (Living Room)
    # Row 3 (y: ~340 to ~485): Kitchen, Additional View (Kitchen), Breakfast/Laundry
    # Row 4 (y: ~485 to ~630): Additional View (Laundry/Hall), Bedroom 1, Bedroom 1 (View 2)
    # Row 5 (y: ~630 to ~770): Bedroom 2, Bedroom 2 (View 2), Bathroom 1
    p13_labels = [
        ("Front_Left_Exterior", "Front/Left Exterior", (35, 65, 235, 205)),
        ("Front_Right_Exterior", "Front/Right Exterior", (235, 65, 400, 205)),
        ("Rear_Left_Exterior", "Rear/Left Exterior", (400, 65, 565, 205)),
        ("Rear_Right_Exterior", "Rear/Right Exterior", (35, 205, 235, 345)),
        ("Living_Room", "Living Room", (235, 205, 400, 345)),
        ("Living_Room_Addl", "Living Room Additional View", (400, 205, 565, 345)),
        ("Kitchen", "Kitchen", (35, 345, 235, 490)),
        ("Kitchen_Addl", "Kitchen Additional View", (235, 345, 400, 490)),
        ("Breakfast_Laundry", "Breakfast/Laundry Area", (400, 345, 565, 490)),
        ("Laundry_Hall_Addl", "Additional View (Interior/Utility)", (35, 490, 235, 630)),
        ("Bedroom_1_View1", "Bedroom 1", (235, 490, 400, 630)),
        ("Bedroom_1_View2", "Bedroom 1 Additional View", (400, 490, 565, 630)),
        ("Bedroom_2_View1", "Bedroom 2", (35, 630, 235, 775)),
        ("Bedroom_2_View2", "Bedroom 2 Additional View", (235, 630, 400, 775)),
        ("Bathroom_1_View1", "Bathroom 1", (400, 630, 565, 775)),
    ]
    for slot, label, box in p13_labels:
        jobs.append({"page": 13, "slot": slot, "label": label, "crop_box": box})

    # Page 14: Photograph Addendum (8 photos)
    # Row 1: Bathroom 1 (View 2), Bedroom 3, Bedroom 3 (View 2)
    # Row 2: Bathroom 2, Bathroom 2 (View 2), Smoke Detector
    # Row 3: Water Heater, Breaker Panel
    p14_labels = [
        ("Bathroom_1_View2", "Bathroom 1 Additional View", (35, 65, 235, 205)),
        ("Bedroom_3_View1", "Bedroom 3", (235, 65, 400, 205)),
        ("Bedroom_3_View2", "Bedroom 3 Additional View", (400, 65, 565, 205)),
        ("Bathroom_2_View1", "Bathroom 2", (35, 205, 235, 345)),
        ("Bathroom_2_View2", "Bathroom 2 Additional View", (235, 205, 400, 345)),
        ("Smoke_Detector", "Smoke Detector", (400, 205, 565, 345)),
        ("Water_Heater", "Water Heater", (35, 345, 235, 490)),
        ("Breaker_Panel", "Electrical Breaker Panel", (235, 345, 400, 490)),
    ]
    for slot, label, box in p14_labels:
        jobs.append({"page": 14, "slot": slot, "label": label, "crop_box": box})

    # Page 15: Building Sketch
    jobs.append({
        "page": 15, "slot": "Building_Sketch", "label": "Building Sketch Floor Plan Layout",
        "crop_box": (60, 150, 550, 680)
    })

    # Page 23: Rental Photo Page (3 rentals)
    jobs.extend([
        {"page": 23, "slot": "Rental_1", "label": "Rental Comparable 1 (1013 Vestal St)", "crop_box": (30, 80, 540, 300)},
        {"page": 23, "slot": "Rental_2", "label": "Rental Comparable 2 (1315 W 16th St)", "crop_box": (30, 300, 540, 520)},
        {"page": 23, "slot": "Rental_3", "label": "Rental Comparable 3 (2015 Franklin St)", "crop_box": (30, 520, 540, 740)},
    ])

    # Page 26: Location Map
    jobs.append({
        "page": 26, "slot": "Location_Map", "label": "Appraisal Location Map with Comparables",
        "crop_box": (60, 90, 550, 750)
    })

    # Page 27: Aerial Map
    jobs.append({
        "page": 27, "slot": "Aerial_Map", "label": "Aerial Map Subject and Neighborhood",
        "crop_box": (60, 90, 550, 750)
    })

    print(f"\nProcessing total of {len(jobs)} appraisal images/features...\n")

    summary_results = []

    for i, job in enumerate(jobs):
        pno = job["page"]
        slot = job["slot"]
        label = job["label"]
        box = job["crop_box"]

        # Render high quality crop
        page = doc[pno - 1]
        rect = pymupdf.Rect(box[0], box[1], box[2], box[3])
        pix = page.get_pixmap(clip=rect, dpi=200)
        img_filename = f"p{pno:02d}_{slot}.png"
        img_path = os.path.join(images_dir, img_filename)
        pix.save(img_path)

        json_filename = f"p{pno:02d}_{slot}_result.json"
        json_path = os.path.join(results_dir, json_filename)

        print(f"[{i+1}/{len(jobs)}] Validating Page {pno} ({slot}): '{label}'...", flush=True)

        try:
            res = process_image(
                image_path=img_path,
                expected_label=label,
                output_json=json_path
            )
            v_status = res["validation"]["status"]
            v_reason = res["validation"]["reason"]
            observed = res.get("gemma", {}).get("observed_object", "N/A")
            conf = res.get("gemma", {}).get("confidence", 0.0)
            print(f"    -> Status: {v_status} | Observed: {observed} (Conf: {conf})", flush=True)
            import time
            time.sleep(0.5)
        except Exception as e:
            print(f"    -> Error processing {slot}: {e}", flush=True)
            res = {
                "image_path": img_path,
                "expected_label": label,
                "error": str(e),
                "validation": {
                    "status": "FAIL",
                    "reason": f"Processing exception: {e}"
                }
            }

        summary_results.append({
            "index": i + 1,
            "page": pno,
            "slot": slot,
            "expected_label": label,
            "image_path": img_path,
            "json_path": json_path,
            "status": res["validation"]["status"],
            "reason": res["validation"]["reason"],
            "observed_object": res.get("gemma", {}).get("observed_object", ""),
            "gemma_confidence": res.get("gemma", {}).get("confidence", 0.0),
            "ocr_text": res.get("paddleocr", {}).get("text", [])
        })

    # Save consolidated summary JSON
    summary_path = os.path.join(output_base_dir, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)

    print(f"\n==========================================", flush=True)
    print(f"Appraisal Image Validation Complete!", flush=True)
    print(f"Summary JSON saved to: {summary_path}", flush=True)
    print(f"==========================================", flush=True)

    # Print summary counts
    pass_cnt = sum(1 for r in summary_results if r["status"] == "PASS")
    review_cnt = sum(1 for r in summary_results if r["status"] == "REVIEW")
    fail_cnt = sum(1 for r in summary_results if r["status"] == "FAIL")
    print(f"Total Images: {len(summary_results)} | PASS: {pass_cnt} | REVIEW: {review_cnt} | FAIL: {fail_cnt}", flush=True)

    return summary_results

if __name__ == "__main__":
    pdf_file = r"C:\Users\Admin\Downloads\1806 W 16th St.pdf"
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    extract_and_validate_pdf(pdf_file)
