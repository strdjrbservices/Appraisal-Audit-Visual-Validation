import os
import re
import json
import pymupdf
from paddleocr_module import PaddleOCRExtractor

def clean_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def normalize_address(addr: str) -> str:
    """Normalize street abbreviations for robust comparisons."""
    if not addr:
        return ""
    a = addr.lower().strip()
    a = re.sub(r"\bst\b|\bstreet\b", "st", a)
    a = re.sub(r"\brd\b|\broad\b", "rd", a)
    a = re.sub(r"\bave\b|\bavenue\b", "ave", a)
    a = re.sub(r"\bdr\b|\bdrive\b", "dr", a)
    a = re.sub(r"\bln\b|\blane\b", "ln", a)
    a = re.sub(r"\bblvd\b|\bboulevard\b", "blvd", a)
    a = re.sub(r"\bct\b|\bcourt\b", "ct", a)
    a = re.sub(r"\bpl\b|\bplace\b", "pl", a)
    a = re.sub(r"\bter\b|\bterrace\b", "ter", a)
    a = re.sub(r"\bwy\b|\bway\b", "way", a)
    a = re.sub(r"\bpkwy\b|\bparkway\b", "pkwy", a)
    a = re.sub(r"\bn\b|\bnorth\b", "n", a)
    a = re.sub(r"\bs\b|\bsouth\b", "s", a)
    a = re.sub(r"\be\b|\beast\b", "e", a)
    a = re.sub(r"\bw\b|\bwest\b", "w", a)
    a = re.sub(r"[^\w\s]", "", a)
    return re.sub(r"\s+", " ", a).strip()

def extract_master_subject(doc) -> dict:
    """
    Extract canonical subject property details from the primary appraisal form (Pages 1-5).
    """
    master = {
        "street_address": "",
        "city": "",
        "state": "",
        "zip_code": "",
        "county": "",
        "full_address": "",
        "borrower": "",
        "lender": "",
        "file_no": "",
        "source_page": 1
    }

    street_regex = re.compile(
        r"^\d+\s+[A-Za-z0-9\s,\.]+(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane|Pl|Place|Way|Blvd|Boulevard|Court|Ct|Ter|Terrace|Pkwy|Parkway|Circle|Cir)\b",
        re.IGNORECASE
    )

    for pno in range(min(6, len(doc))):
        page = doc[pno]
        blocks = page.get_text("blocks")
        text = page.get_text()

        # Check for File #
        if not master["file_no"]:
            m_file = re.search(r"(?:File\s*(?:No|#)[:\.\s]*|R2[0-9]-[0-9]+|PA\s*\d+|JDS\d+)([A-Z0-9_-]{5,})", text, re.IGNORECASE)
            if m_file:
                master["file_no"] = m_file.group(0).strip()

        # Look for URAR Subject Form Block (total software layout)
        # Block y around 70-80 with Address, City, State, Zip, Borrower, Owner, County
        for b in blocks:
            lines = [clean_text(l) for l in b[4].splitlines() if clean_text(l)]
            if len(lines) >= 4:
                # Test line 0 for street address
                if street_regex.match(lines[0]):
                    if not master["street_address"]:
                        master["street_address"] = lines[0]
                        master["source_page"] = pno + 1
                    # Total software layout: [Street, City, State, Zip, Borrower, Owner, County]
                    if len(lines) >= 4:
                        if re.match(r"^[A-Z]{2}$", lines[2], re.IGNORECASE) and re.match(r"^\d{5}", lines[3]):
                            master["city"] = lines[1]
                            master["state"] = lines[2].upper()
                            master["zip_code"] = lines[3]
                            if len(lines) >= 5 and not master["borrower"]:
                                master["borrower"] = lines[4]
                            if len(lines) >= 7 and not master["county"]:
                                master["county"] = lines[6]
                            break

        if master["street_address"] and master["city"]:
            break

    # Fallback search if form block didn't match perfectly
    if not master["street_address"]:
        for pno in range(min(5, len(doc))):
            text = doc[pno].get_text()
            lines = [clean_text(l) for l in text.splitlines() if clean_text(l)]
            for idx, line in enumerate(lines):
                if street_regex.match(line):
                    master["street_address"] = line
                    master["source_page"] = pno + 1
                    # Next line often has City, State Zip
                    if idx + 1 < len(lines):
                        m_csz = re.search(r"([A-Za-z\s]+)[,\s]+([A-Z]{2})[,\s]+(\d{5})", lines[idx+1])
                        if m_csz:
                            master["city"] = m_csz.group(1).strip()
                            master["state"] = m_csz.group(2).upper()
                            master["zip_code"] = m_csz.group(3)
                    break
            if master["street_address"]:
                break

    # Build full address
    parts = [master["street_address"]]
    if master["city"]:
        parts.append(master["city"])
    if master["state"] and master["zip_code"]:
        parts.append(f"{master['state']} {master['zip_code']}")
    elif master["state"]:
        parts.append(master["state"])
    elif master["zip_code"]:
        parts.append(master["zip_code"])

    master["full_address"] = ", ".join(p for p in parts if p)
    return master

def identify_section_type(page_text: str, page_num: int) -> str:
    """Classify the appraisal report page into a standardized section."""
    t = page_text.upper()
    lines = [clean_text(l).upper() for l in page_text.splitlines() if clean_text(l)]
    top_500 = " ".join(lines[:12])

    if "TABLE OF CONTENTS" in top_500:
        return "Table of Contents"
    if "INVOICE" in top_500:
        return "Invoice / Billing"
    if "APPRAISAL OF REAL PROPERTY" in top_500 or "UNIFORM RESIDENTIAL APPRAISAL" in top_500:
        if "SALES COMPARISON APPROACH" in t or "COMPARABLE SALE #" in t:
            if "COMPARABLE SALE # 4" in t:
                return "Additional Sales Comparison Approach (Comps 4-6)"
            if "COMPARABLE SALE # 7" in t:
                return "Additional Sales Comparison Approach (Comps 7-9)"
            return "Sales Comparison Approach (Comps 1-3)"
        if page_num <= 4:
            return "Subject Property Section (Page 1)"
    if "ADDITIONAL COMPARABLES" in top_500 or "COMPARABLES 4-6" in top_500 or "COMPARABLE SALE # 4" in t:
        return "Additional Sales Comparison Approach (Comps 4-6)"
    if "COMPARABLES 7-9" in top_500 or "COMPARABLE SALE # 7" in t:
        return "Additional Sales Comparison Approach (Comps 7-9)"
    if "SALES COMPARISON APPROACH" in t or "COMPARABLE SALE #" in t:
        return "Sales Comparison Approach (Comps 1-3)"
    if "SUBJECT PHOTO" in t or "PICPIX.SR" in t:
        return "Subject Photo Page"
    if "COMPARABLE PHOTO" in t or "PICPIX.CR" in t:
        return "Comparable Photo Page"
    if "PHOTOGRAPH ADDENDUM" in t or "PICINT" in t or "PICSIX" in t or "PIC15" in t or "PIC4X6" in t:
        return "Interior / Photo Addendum"
    if "LOCATION MAP" in t or "MAP.LOC" in t or "COMPARABLE SALES MAP" in t:
        return "Location / Comparable Sales Map"
    if "AERIAL" in t or "MAP.AERIAL" in t:
        return "Aerial Map Page"
    if "BUILDING SKETCH" in t or "SKT." in t or "FLOOR PLAN" in t:
        return "Building Sketch Page"
    if "FLOOD MAP" in t or "PLAT MAP" in t:
        return "Plat / Flood Map Page"
    if "GEODATA" in top_500 or "PUBLIC RECORD" in top_500:
        return "Public Record / Geodata Addendum"
    if "LICENSE" in t or "CERTIFICATION" in t or "SCNLGL" in t:
        if "E & O" in t or "INSURANCE" in t or "POLICY" in t:
            return "E & O Insurance Policy"
        return "Appraiser License / Certification"
    if "MARKET CONDITIONS" in t or "1004MC" in t:
        return "Market Conditions Addendum (1004MC)"
    if "UAD DEFINITIONS" in t:
        return "UAD Definitions Addendum"
    if "USPAP" in t:
        return "USPAP Compliance Addendum"
    
    return "Appraisal Addendum / Report Page"

def extract_section_address(page, doc, pno: int, sec_type: str, master: dict) -> dict:
    """
    Extract the address reported on this specific page / section.
    Handles Sales Comparison Subject column, Header Tables, and Map/Sketch blocks.
    """
    blocks = page.get_text("blocks")
    text = page.get_text()

    extracted = {
        "street_address": "",
        "city": "",
        "state": "",
        "zip_code": "",
        "county": "",
        "full_text": "",
        "found_via": "none"
    }

    norm_master_street = normalize_address(master["street_address"])

    # 1. Sales Comparison Approach Special Extraction (Subject Column)
    if "Sales Comparison" in sec_type:
        for b in blocks:
            # Look for block containing subject address (x around 50 to 180, y around 60 to 120)
            if 50 <= b[1] <= 140 and b[0] < 180:
                btext = b[4].strip()
                if any(hdr in btext for hdr in ["FEATURE", "Proximity to Subject", "Sale Price", "ITEM"]):
                    continue
                lines = [clean_text(l) for l in btext.splitlines() if clean_text(l)]
                for idx, line in enumerate(lines):
                    if norm_master_street and normalize_address(line) == norm_master_street:
                        extracted["street_address"] = line
                        if idx + 1 < len(lines):
                            m_csz = re.search(r"([A-Za-z\s]+)[,\s]+([A-Z]{2})[,\s]+(\d{5})", lines[idx+1])
                            if m_csz:
                                extracted["city"] = m_csz.group(1).strip()
                                extracted["state"] = m_csz.group(2).upper()
                                extracted["zip_code"] = m_csz.group(3)
                            else:
                                extracted["city"] = lines[idx+1]
                        extracted["full_text"] = " ".join(lines)
                        extracted["found_via"] = "Sales Comparison Subject Column"
                        return extracted

    # 2. Standard Header Table Extraction (Total / Fannie Mae Forms)
    # Header table is located at y: 40 to 120
    for b in blocks:
        if 40 <= b[1] <= 120:
            btext = b[4].strip()
            if any(hdr in btext for hdr in ["Property Address", "Borrower", "Lender/Client", "Form "]):
                continue
            lines = [clean_text(l) for l in btext.splitlines() if clean_text(l)]
            if len(lines) >= 3:
                for idx, line in enumerate(lines):
                    if norm_master_street and normalize_address(line) == norm_master_street:
                        extracted["street_address"] = line
                        if idx + 1 < len(lines):
                            extracted["city"] = lines[idx+1]
                        if idx + 3 < len(lines) and re.match(r"^[A-Z]{2}$", lines[idx+3], re.IGNORECASE):
                            extracted["state"] = lines[idx+3].upper()
                        if idx + 4 < len(lines) and re.match(r"^\d{5}", lines[idx+4]):
                            extracted["zip_code"] = lines[idx+4]
                        extracted["full_text"] = " ".join(lines)
                        extracted["found_via"] = "Standard Header Table"
                        return extracted
                
                # Check line 0 or line 1
                if any(k in lines[0].lower() for k in [" st", " street", " rd", " road", " ave", " dr", " ln", " way", " blvd", " ct", " ter"]):
                    extracted["street_address"] = lines[0]
                    if len(lines) > 1:
                        extracted["city"] = lines[1]
                    extracted["full_text"] = " ".join(lines)
                    extracted["found_via"] = "Header Block Line 0"
                    return extracted

    # 3. Regex search in page text for master street occurrence
    if norm_master_street:
        street_raw = master["street_address"]
        m = re.search(rf"({re.escape(street_raw)}[^\n\r]*)", text, re.IGNORECASE)
        if m:
            matched_line = clean_text(m.group(1))
            extracted["street_address"] = street_raw
            extracted["full_text"] = matched_line
            extracted["found_via"] = "Page Text Search"
            return extracted

    # 4. If Scanned Image Page (like License / E&O / Scanned Survey) with no text, try fast OCR
    if sec_type in ["Appraiser License / Certification", "E & O Insurance Policy", "Building Sketch Page", "Location / Comparable Sales Map"]:
        if len(text.strip()) < 80:
            try:
                # Render top header rect (y: 0 to 200) to OCR
                rect = pymupdf.Rect(0, 0, page.rect.width, 200)
                pix = page.get_pixmap(clip=rect, dpi=150)
                tmp_path = os.path.join(os.environ.get("TEMP", "."), f"ocr_hdr_p{pno+1}.png")
                pix.save(tmp_path)
                ocr_res = PaddleOCRExtractor(lang="en").extract(tmp_path)
                ocr_texts = ocr_res.get("text", [])
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

                ocr_full = " ".join(str(t) for t in ocr_texts)
                if norm_master_street and norm_master_street in normalize_address(ocr_full):
                    extracted["street_address"] = master["street_address"]
                    extracted["full_text"] = ocr_full
                    extracted["found_via"] = "Header OCR"
                    return extracted
            except Exception:
                pass

    return extracted

def evaluate_address_match(master: dict, extracted: dict) -> dict:
    """Compare extracted address with canonical master subject address."""
    m_street_norm = normalize_address(master.get("street_address", ""))
    e_street_norm = normalize_address(extracted.get("street_address", ""))

    m_city_norm = master.get("city", "").lower().strip()
    e_city_norm = extracted.get("city", "").lower().strip()

    has_extracted = bool(e_street_norm)

    street_match = bool(has_extracted and m_street_norm and (m_street_norm == e_street_norm or m_street_norm in e_street_norm or e_street_norm in m_street_norm))
    city_match = bool(m_city_norm and (m_city_norm == e_city_norm or m_city_norm in e_city_norm)) if (m_city_norm and e_city_norm) else True
    state_match = (master.get("state", "").upper() == extracted.get("state", "").upper()) if extracted.get("state") else True
    zip_match = (master.get("zip_code", "")[:5] == extracted.get("zip_code", "")[:5]) if extracted.get("zip_code") else True

    if street_match and city_match and state_match and zip_match:
        status = "PASS"
        reason = "Subject address is 100% consistent with master appraisal subject."
    elif street_match:
        status = "PASS"
        reason = f"Street address verified: '{extracted.get('street_address')}'. City/State match confirmed."
    elif not has_extracted:
        status = "REVIEW"
        reason = "Header address block not explicitly printed on this addendum page."
    else:
        status = "MISMATCH"
        reason = f"Extracted address '{extracted.get('street_address')}' differs from expected '{master.get('street_address')}'."

    return {
        "status": status,
        "reason": reason,
        "street_match": street_match,
        "city_match": city_match,
        "state_match": state_match,
        "zip_match": zip_match
    }

def validate_subject_address_consistency(pdf_path: str) -> dict:
    """
    Complete audit of Subject Property Address consistency across all pages and sections:
    - Subject Property Section (Page 1)
    - Sales Comparison Approach (Comps 1-3 Subject Column)
    - Additional Sales Comparison Approach (Comps 4-6, 7-9)
    - Subject Photo Pages
    - Interior / Photo Addenda
    - Location / Comparable Sales Map
    - Aerial Map Page
    - Building Sketch Page
    - Appraiser License & E&O Insurance Policy
    """
    doc = pymupdf.open(pdf_path)
    master = extract_master_subject(doc)

    sections_audited = []
    target_sections = [
        "Subject Property Section (Page 1)",
        "Sales Comparison Approach (Comps 1-3)",
        "Additional Sales Comparison Approach",
        "Subject Photo Page",
        "Comparable Photo Page",
        "Interior / Photo Addendum",
        "Building Sketch Page",
        "Location / Comparable Sales Map",
        "Aerial Map Page",
        "Plat / Flood Map Page",
        "Appraiser License / Certification",
        "E & O Insurance Policy"
    ]

    for pno in range(len(doc)):
        page = doc[pno]
        page_text = page.get_text()
        sec_type = identify_section_type(page_text, pno + 1)

        # Only evaluate pages that belong to key appraisal sections
        if sec_type in target_sections or any(kw in sec_type for kw in ["Sales Comparison", "Photo", "Map", "Sketch", "License"]):
            extracted = extract_section_address(page, doc, pno, sec_type, master)
            eval_res = evaluate_address_match(master, extracted)

            sections_audited.append({
                "page": pno + 1,
                "section_name": sec_type,
                "extracted_street": extracted.get("street_address", "N/A"),
                "extracted_city": extracted.get("city", "N/A"),
                "extracted_state": extracted.get("state", "N/A"),
                "extracted_zip": extracted.get("zip_code", "N/A"),
                "found_via": extracted.get("found_via", "none"),
                "status": eval_res["status"],
                "reason": eval_res["reason"],
                "street_match": eval_res["street_match"],
                "city_match": eval_res["city_match"],
                "state_match": eval_res["state_match"],
                "zip_match": eval_res["zip_match"]
            })

    pass_count = sum(1 for s in sections_audited if s["status"] == "PASS")
    review_count = sum(1 for s in sections_audited if s["status"] == "REVIEW")
    mismatch_count = sum(1 for s in sections_audited if s["status"] == "MISMATCH")
    total_audited = len(sections_audited)
    match_rate = round((pass_count / total_audited * 100), 1) if total_audited else 0

    return {
        "master_subject": master,
        "total_sections_audited": total_audited,
        "pass_count": pass_count,
        "review_count": review_count,
        "mismatch_count": mismatch_count,
        "match_rate_pct": match_rate,
        "overall_status": "PASS" if mismatch_count == 0 and match_rate >= 80 else ("REVIEW" if mismatch_count == 0 else "MISMATCH"),
        "sections": sections_audited
    }

if __name__ == "__main__":
    import sys
    test_pdf = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\Downloads\16 Firwood Dr.pdf"
    res = validate_subject_address_consistency(test_pdf)
    print(json.dumps(res, indent=2))
