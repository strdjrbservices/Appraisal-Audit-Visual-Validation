import os
import sys
import json
import base64
import pymupdf
from PIL import Image
import re
from pipeline import process_image
from address_validator import validate_subject_address_consistency, extract_master_subject

def extract_pdf_metadata(doc):
    master = extract_master_subject(doc)
    metadata = {
        "property_address": master.get("full_address") or master.get("street_address") or "Appraisal Subject Property",
        "street_address": master.get("street_address", ""),
        "city": master.get("city", ""),
        "state": master.get("state", ""),
        "zip_code": master.get("zip_code", ""),
        "county": master.get("county", ""),
        "borrower": master.get("borrower") or "N/A",
        "lender": master.get("lender") or "N/A",
        "file_no": master.get("file_no") or "N/A",
        "total_pages": len(doc)
    }
    return metadata

def parse_photo_slots_from_page(page, page_num):
    text = page.get_text()
    blocks = page.get_text("blocks")
    images = page.get_image_info(xrefs=True)
    
    # Filter real images (width > 60, height > 60)
    real_images = [img for img in images if (img["bbox"][2] - img["bbox"][0]) > 60 and (img["bbox"][3] - img["bbox"][1]) > 60]
    
    # Exclude non-photo pages unless they contain sketches or maps
    page_header = text[:300].upper()
    is_photo_page = any(kw in page_header for kw in [
        "PHOTO", "PHOTOGRAPH", "SUBJECT PHOTO", "COMPARABLE PHOTO", 
        "SKETCH", "LOCATION MAP", "AERIAL", "FLOOR PLAN", "RENTAL PHOTO",
        "BUILDING SKETCH", "PLAT MAP", "FLOOD MAP"
    ]) or any(f in text for f in ["PICSIX", "PICINT", "PIC4X6", "PICPIX", "PIC15", "SKT.BLDSKI", "MAP.LOC", "MAP.AERIAL"])
    
    if not is_photo_page:
        return []
    
    # Check for Form PICSIX2 / PICINT6 (6-photo grid with 2 cols, 3 rows)
    if "PICSIX" in text.upper() or len(real_images) == 6:
        sorted_imgs = sorted(real_images, key=lambda img: (round(img["bbox"][1] / 150) * 150, img["bbox"][0]))
        label_pairs = []
        for b in sorted(blocks, key=lambda b: b[1]):
            if b[1] < 100 or b[1] > 950:
                continue
            btext = b[4].strip()
            if any(h in btext for h in ["Borrower", "Lender", "Property Address", "TOTAL", "Form ", "Centereach", "Suffolk"]):
                continue
            lines = [l.strip() for l in btext.split("\n") if l.strip()]
            if lines:
                label_pairs.append((b, lines))
        
        slots = []
        for idx, img in enumerate(sorted_imgs):
            bbox = img["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            
            best_lbl = None
            for b, lines in label_pairs:
                if bbox[3] - 20 <= b[1] <= bbox[3] + 80:
                    if len(lines) == 2:
                        best_lbl = lines[0] if cx < page.rect.width / 2 else lines[1]
                        break
                    elif len(lines) == 1:
                        best_lbl = lines[0]
                        break
            
            if not best_lbl:
                best_lbl = f"Addendum Photo {idx+1}"
                
            crop_box = (max(0, bbox[0]-10), max(0, bbox[1]-10), min(page.rect.width, bbox[2]+10), min(page.rect.height, bbox[3]+10))
            slots.append({
                "slot": f"Photo_{idx+1}",
                "label": best_lbl,
                "crop_box": crop_box,
                "page": page_num
            })
        return slots

    # Special case: 15-photo addendum (Form PIC15 or 3x5 grid)
    if len(real_images) >= 12:
        sorted_imgs = sorted(real_images, key=lambda img: (round(img["bbox"][1] / 100) * 100, img["bbox"][0]))
        text_labels = []
        for b in blocks:
            lines = [l.strip() for l in b[4].split("\n") if l.strip()]
            for l in lines:
                if any(term in l.upper() for term in ["EXTERIOR", "LIVING", "KITCHEN", "BEDROOM", "BATH", "LAUNDRY", "ROOM", "ENTRY", "VIEW", "PANEL", "HEATER", "DETECTOR", "BASEMENT", "ATTIC"]):
                    if len(l) < 50 and not "FORM" in l.upper():
                        text_labels.append(l)
        
        slots = []
        for idx, img in enumerate(sorted_imgs):
            label = text_labels[idx] if idx < len(text_labels) else f"Photo Addendum Slot {idx+1}"
            bbox = img["bbox"]
            crop_box = (max(0, bbox[0]-5), max(0, bbox[1]-5), min(page.rect.width, bbox[2]+5), min(page.rect.height, bbox[3]+5))
            slots.append({
                "slot": f"Addendum_Photo_{idx+1}",
                "label": label,
                "crop_box": crop_box,
                "page": page_num
            })
        return slots

    # Standard 3 vertical photos (Subject Front/Rear/Street or Comparable Sale 1/2/3)
    if 1 <= len(real_images) <= 5:
        sorted_imgs = sorted(real_images, key=lambda img: img["bbox"][1])
        slots = []
        
        for idx, img in enumerate(sorted_imgs):
            bbox = img["bbox"]
            best_label = None
            
            for b in blocks:
                if b[1] < 100 or b[1] > 950:
                    continue
                btext = b[4].strip()
                if not btext or any(h in btext for h in ["Borrower", "Lender", "Property Address", "TOTAL", "Form "]):
                    continue
                
                if abs(b[1] - bbox[1]) < 60 or (bbox[1] <= b[1] <= bbox[3]):
                    first_line = btext.split("\n")[0].strip()
                    if any(k in first_line.upper() for k in ["SUBJECT", "COMPARABLE", "COMP", "FRONT", "REAR", "STREET", "SIDE"]):
                        best_label = first_line
                        break
            
            if not best_label:
                if "SUBJECT" in text.upper():
                    pos_names = ["Subject Front", "Subject Rear", "Subject Street"]
                    best_label = pos_names[idx] if idx < len(pos_names) else f"Subject Photo {idx+1}"
                elif "COMPARABLE" in text.upper():
                    comp_start = 1
                    m_comp = re.search(r"Comparable\s*(\d+)", text, re.IGNORECASE)
                    if m_comp:
                        comp_start = int(m_comp.group(1))
                    best_label = f"Comparable {comp_start + idx}"
                elif "SKETCH" in text.upper():
                    best_label = "Building Sketch Floor Plan Layout"
                elif "AERIAL" in text.upper():
                    best_label = "Aerial Map"
                elif "LOCATION MAP" in text.upper() or "MAP" in text.upper():
                    best_label = "Location Map"
                else:
                    best_label = f"Appraisal Asset {idx+1}"
            
            crop_box = (max(0, bbox[0]-10), max(0, bbox[1]-10), min(page.rect.width, bbox[2]+10), min(page.rect.height, bbox[3]+10))
            slots.append({
                "slot": f"Photo_{idx+1}",
                "label": best_label,
                "crop_box": crop_box,
                "page": page_num
            })
        return slots

    # Page with sketch or map without separate image object (vector graphics)
    if any(kw in text.upper() for kw in ["BUILDING SKETCH", "FLOOR PLAN"]):
        return [{
            "slot": "Building_Sketch",
            "label": "Building Sketch Floor Plan Layout",
            "crop_box": (30, 80, page.rect.width - 30, page.rect.height - 80),
            "page": page_num
        }]

    return []

def generate_html_report(metadata, results, output_html_path, address_audit=None):
    pass_cnt = sum(1 for r in results if r["status"] == "PASS")
    review_cnt = sum(1 for r in results if r["status"] == "REVIEW")
    fail_cnt = sum(1 for r in results if r["status"] == "FAIL")
    pass_rate = (pass_cnt / len(results) * 100) if results else 0
    avg_conf = (sum(r.get("gemma_confidence", 0) for r in results) / len(results) * 100) if results else 0

    # Address audit stats
    addr_total = address_audit.get("total_sections_audited", 0) if address_audit else 0
    addr_pass = address_audit.get("pass_count", 0) if address_audit else 0
    addr_review = address_audit.get("review_count", 0) if address_audit else 0
    addr_mismatch = address_audit.get("mismatch_count", 0) if address_audit else 0
    addr_rate = address_audit.get("match_rate_pct", 100) if address_audit else 100
    addr_status = address_audit.get("overall_status", "PASS") if address_audit else "PASS"

    # Address Audit Rows HTML
    addr_rows_html = ""
    if address_audit and address_audit.get("sections"):
        for sec in address_audit["sections"]:
            s_status = sec["status"]
            s_color = "#10b981" if s_status == "PASS" else ("#f59e0b" if s_status == "REVIEW" else "#ef4444")
            s_bg = "rgba(16, 185, 129, 0.15)" if s_status == "PASS" else ("rgba(245, 158, 11, 0.15)" if s_status == "REVIEW" else "rgba(239, 68, 68, 0.15)")

            addr_rows_html += f"""
            <tr class="audit-row" data-status="{s_status}">
                <td><span class="page-tag">Page {sec['page']}</span></td>
                <td><strong>{sec['section_name']}</strong></td>
                <td><span class="addr-text">{sec.get('extracted_street') or '—'}</span></td>
                <td>{sec.get('extracted_city') or '—'}</td>
                <td>{sec.get('extracted_state') or '—'}</td>
                <td>{sec.get('extracted_zip') or '—'}</td>
                <td><span class="badge-inline" style="background: {s_bg}; color: {s_color}; border: 1px solid {s_color};">{s_status}</span></td>
                <td class="reason-cell">{sec.get('reason', '')}</td>
            </tr>
            """

    # Build card items HTML
    cards_html = ""
    for r in results:
        status = r["status"]
        status_color = "#10b981" if status == "PASS" else ("#f59e0b" if status == "REVIEW" else "#ef4444")
        badge_bg = "rgba(16, 185, 129, 0.15)" if status == "PASS" else ("rgba(245, 158, 11, 0.15)" if status == "REVIEW" else "rgba(239, 68, 68, 0.15)")
        conf_pct = int(r.get("gemma_confidence", 0) * 100)
        
        # Read image to base64 for standalone HTML embedding
        img_b64 = ""
        if os.path.exists(r["image_path"]):
            with open(r["image_path"], "rb") as f:
                img_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")

        ocr_tags = "".join([f'<span class="ocr-tag">{t}</span>' for t in r.get("ocr_text", [])[:8]])
        if not ocr_tags:
            ocr_tags = '<span class="ocr-tag ocr-empty">No overlay text detected</span>'

        cards_html += f"""
        <div class="card" data-status="{status}">
            <div class="card-img-wrap">
                <img src="{img_b64}" alt="{r['expected_label']}" loading="lazy" onclick="openModal('{img_b64}', '{r['expected_label']}')" />
                <span class="badge" style="background: {badge_bg}; color: {status_color}; border: 1px solid {status_color};">{status}</span>
            </div>
            <div class="card-body">
                <div class="card-header-row">
                    <span class="page-tag">Page {r['page']}</span>
                    <span class="conf-text">{conf_pct}% Confidence</span>
                </div>
                <h3 class="card-title">{r['expected_label']}</h3>
                
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width: {conf_pct}%; background: {status_color};"></div>
                </div>

                <div class="section-label">Observed Verification</div>
                <p class="observed-text">{r.get('observed_object', 'N/A')}</p>

                <div class="section-label">OCR Evidence Tokens</div>
                <div class="ocr-tags-wrap">{ocr_tags}</div>

                <div class="reason-box">
                    <strong>Rule Engine:</strong> {r.get('reason', '')}
                </div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Appraisal Validation Report - {metadata['property_address']}</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-card: #172033;
            --border: #1f2d48;
            --primary: #38bdf8;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --pass: #10b981;
            --review: #f59e0b;
            --fail: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 32px 24px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        
        .header {{
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(17, 24, 39, 0.9));
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 28px;
            backdrop-filter: blur(16px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .header-title h1 {{
            font-size: 26px;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        }}
        .header-subtitle {{ color: var(--primary); font-weight: 600; font-size: 15px; }}
        .header-meta {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }}
        .meta-pill {{
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 13px;
        }}
        .meta-pill strong {{ color: #cbd5e1; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .stat-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }}
        .stat-card::after {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 3px;
        }}
        .stat-card.stat-total::after {{ background: var(--primary); }}
        .stat-card.stat-pass::after {{ background: var(--pass); }}
        .stat-card.stat-review::after {{ background: var(--review); }}
        .stat-card.stat-fail::after {{ background: var(--fail); }}
        
        .stat-label {{ font-size: 12px; color: var(--text-muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 30px; font-weight: 800; margin: 6px 0 2px; }}
        .stat-sub {{ font-size: 13px; color: var(--text-muted); }}

        /* Address Consistency Panel */
        .audit-panel {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 32px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .audit-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .audit-title h2 {{
            font-size: 20px;
            font-weight: 800;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .audit-summary-badge {{
            font-size: 13px;
            font-weight: 700;
            padding: 6px 14px;
            border-radius: 20px;
            background: rgba(16, 185, 129, 0.15);
            color: var(--pass);
            border: 1px solid var(--pass);
        }}
        .master-addr-card {{
            background: rgba(56, 189, 248, 0.06);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .master-addr-card .label {{ font-size: 12px; color: var(--primary); font-weight: 700; text-transform: uppercase; }}
        .master-addr-card .addr-val {{ font-size: 16px; font-weight: 800; color: #fff; margin-top: 2px; }}
        
        .audit-table-wrap {{
            overflow-x: auto;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .audit-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        .audit-table th {{
            background: #1e293b;
            color: var(--text-muted);
            font-weight: 700;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.5px;
        }}
        .audit-table td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text);
        }}
        .audit-table tr:last-child td {{ border-bottom: none; }}
        .audit-table tr:hover {{ background: rgba(255,255,255,0.02); }}
        .badge-inline {{
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 6px;
            display: inline-block;
        }}
        .reason-cell {{ font-size: 12px; color: var(--text-muted); max-width: 320px; }}

        .section-heading {{
            font-size: 20px;
            font-weight: 800;
            color: #fff;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .toolbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }}
        .filters {{
            display: flex;
            gap: 8px;
            background: var(--surface);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        .filter-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .filter-btn.active {{
            background: var(--primary);
            color: #0f172a;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .card:hover {{
            transform: translateY(-4px);
            border-color: rgba(56, 189, 248, 0.4);
        }}
        .card-img-wrap {{
            position: relative;
            width: 100%;
            height: 230px;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }}
        .card-img-wrap img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            cursor: zoom-in;
            transition: transform 0.3s;
        }}
        .card-img-wrap img:hover {{ transform: scale(1.03); }}
        
        .badge {{
            position: absolute;
            top: 12px;
            right: 12px;
            font-size: 12px;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 8px;
            letter-spacing: 0.5px;
            backdrop-filter: blur(8px);
        }}
        .card-body {{
            padding: 18px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }}
        .card-header-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .page-tag {{
            font-size: 11px;
            font-weight: 700;
            color: var(--primary);
            background: rgba(56, 189, 248, 0.1);
            padding: 2px 8px;
            border-radius: 6px;
        }}
        .conf-text {{ font-size: 12px; color: var(--text-muted); font-weight: 600; }}
        .card-title {{ font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #fff; }}
        
        .conf-bar-bg {{
            height: 4px;
            background: rgba(255,255,255,0.06);
            border-radius: 2px;
            margin-bottom: 14px;
            overflow: hidden;
        }}
        .conf-bar-fill {{ height: 100%; border-radius: 2px; }}

        .section-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }}
        .observed-text {{ font-size: 13px; color: #cbd5e1; margin-bottom: 12px; min-height: 38px; }}
        
        .ocr-tags-wrap {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 14px; min-height: 24px; }}
        .ocr-tag {{
            font-size: 11px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 2px 6px;
            border-radius: 4px;
            color: #94a3b8;
        }}
        .ocr-empty {{ font-style: italic; opacity: 0.6; }}

        .reason-box {{
            margin-top: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            color: #94a3b8;
            border-left: 3px solid var(--border);
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            backdrop-filter: blur(10px);
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .modal.active {{ display: flex; }}
        .modal-content {{
            max-width: 90%;
            max-height: 90%;
            border-radius: 12px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <div class="header-title">
                <h1>Appraisal Audit & Visual Validation Report</h1>
                <div class="header-subtitle">{metadata['property_address']}</div>
            </div>
            <div class="header-meta">
                <div class="meta-pill"><strong>File No:</strong> {metadata['file_no']}</div>
                <div class="meta-pill"><strong>Borrower:</strong> {metadata['borrower']}</div>
                <div class="meta-pill"><strong>Lender:</strong> {metadata['lender']}</div>
                <div class="meta-pill"><strong>Pages:</strong> {metadata['total_pages']}</div>
            </div>
        </header>

        <!-- Section 1: Subject Address Consistency Audit -->
        <section class="audit-panel">
            <div class="audit-header">
                <div class="audit-title">
                    <h2>📍 Subject Address Consistency Audit</h2>
                </div>
                <div class="audit-summary-badge">
                    {addr_pass} of {addr_total} Sections Verified ({addr_rate:.0f}% Pass Rate)
                </div>
            </div>

            <div class="master-addr-card">
                <div>
                    <div class="label">Canonical Subject Property Address</div>
                    <div class="addr-val">{metadata['property_address']}</div>
                </div>
                <div style="display: flex; gap: 10px;">
                    <div class="meta-pill"><strong>County:</strong> {metadata.get('county') or 'N/A'}</div>
                    <div class="meta-pill"><strong>Audit Status:</strong> <span style="color: {('#10b981' if addr_status=='PASS' else '#f59e0b')}; font-weight:800;">{addr_status}</span></div>
                </div>
            </div>

            <div class="audit-table-wrap">
                <table class="audit-table">
                    <thead>
                        <tr>
                            <th>Page</th>
                            <th>Section / Addendum Category</th>
                            <th>Extracted Street Address</th>
                            <th>City</th>
                            <th>State</th>
                            <th>Zip</th>
                            <th>Status</th>
                            <th>Audit Verification Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {addr_rows_html}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Section 2: Visual & Image Validation Summary -->
        <div class="section-heading">
            🖼️ Photo & Visual Asset Verification
        </div>

        <section class="stats-grid">
            <div class="stat-card stat-total">
                <div class="stat-label">Total Assets Analyzed</div>
                <div class="stat-value" style="color: var(--primary);">{len(results)}</div>
                <div class="stat-sub">Across {metadata['total_pages']} appraisal pages</div>
            </div>
            <div class="stat-card stat-pass">
                <div class="stat-label">Visual Pass Rate</div>
                <div class="stat-value" style="color: var(--pass);">{pass_rate:.0f}%</div>
                <div class="stat-sub">{pass_cnt} Verified Matches</div>
            </div>
            <div class="stat-card stat-review">
                <div class="stat-label">Review Flags</div>
                <div class="stat-value" style="color: var(--review);">{review_cnt}</div>
                <div class="stat-sub">Requires manual check</div>
            </div>
            <div class="stat-card stat-fail">
                <div class="stat-label">Average Confidence</div>
                <div class="stat-value" style="color: #a855f7;">{avg_conf:.0f}%</div>
                <div class="stat-sub">Vision model consensus</div>
            </div>
        </section>

        <div class="toolbar">
            <div class="filters">
                <button class="filter-btn active" onclick="filterStatus('all')">All Assets ({len(results)})</button>
                <button class="filter-btn" onclick="filterStatus('PASS')">Passed ({pass_cnt})</button>
                <button class="filter-btn" onclick="filterStatus('REVIEW')">Review ({review_cnt})</button>
                <button class="filter-btn" onclick="filterStatus('FAIL')">Failed ({fail_cnt})</button>
            </div>
        </div>

        <main class="grid">
            {cards_html}
        </main>
    </div>

    <div id="imageModal" class="modal" onclick="closeModal()">
        <img id="modalImg" class="modal-content" src="" />
    </div>

    <script>
        function filterStatus(status) {{
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            document.querySelectorAll('.card').forEach(card => {{
                if (status === 'all' || card.getAttribute('data-status') === status) {{
                    card.style.display = 'flex';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}

        function openModal(src, title) {{
            const modal = document.getElementById('imageModal');
            const img = document.getElementById('modalImg');
            img.src = src;
            modal.classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('imageModal').classList.remove('active');
        }}
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated Interactive HTML Report: {output_html_path}", flush=True)

def process_appraisal_file(pdf_path, output_dir=None):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if output_dir is None:
        output_dir = os.path.join("validation_output", base_name.replace(" ", "_"))
    
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    results_dir = os.path.join(output_dir, "json_results")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    metadata = extract_pdf_metadata(doc)
    
    # 1. Run Subject Address Consistency Audit
    print(f"\n=======================================================")
    print(f"Auditing Appraisal: {os.path.basename(pdf_path)}")
    print(f"Canonical Subject: {metadata['property_address']}")
    print(f"Total Pages: {len(doc)}")
    print(f"=======================================================\n")

    address_audit = validate_subject_address_consistency(pdf_path)
    print(f"[Address Audit] {address_audit['pass_count']}/{address_audit['total_sections_audited']} sections verified ({address_audit['match_rate_pct']}%) | Status: {address_audit['overall_status']}\n")

    # 2. Collect all photo tasks across all pages
    tasks = []
    for pno in range(1, len(doc) + 1):
        page = doc[pno - 1]
        page_slots = parse_photo_slots_from_page(page, pno)
        tasks.extend(page_slots)

    print(f"Detected {len(tasks)} appraisal visual elements to validate.\n")

    results = []
    for i, task in enumerate(tasks):
        pno = task["page"]
        slot = task["slot"]
        label = task["label"]
        box = task["crop_box"]

        # Render crop
        page = doc[pno - 1]
        rect = pymupdf.Rect(box[0], box[1], box[2], box[3])
        pix = page.get_pixmap(clip=rect, dpi=180)
        
        img_name = f"p{pno:02d}_{slot}.png"
        img_path = os.path.join(images_dir, img_name)
        pix.save(img_path)

        json_name = f"p{pno:02d}_{slot}_result.json"
        json_path = os.path.join(results_dir, json_name)

        print(f"[{i+1}/{len(tasks)}] Validating Page {pno} - '{label}'...", flush=True)

        try:
            res = process_image(
                image_path=img_path,
                expected_label=label,
                output_json=json_path
            )
            v_status = res["validation"]["status"]
            observed = res.get("gemma", {}).get("observed_object", "N/A")
            conf = res.get("gemma", {}).get("confidence", 0.0)
            print(f"    -> Status: {v_status} | Observed: {observed[:60]}... (Conf: {conf})", flush=True)
        except Exception as e:
            print(f"    -> Validation Error: {e}", flush=True)
            res = {
                "image_path": img_path,
                "expected_label": label,
                "validation": {
                    "status": "FAIL",
                    "reason": f"Processing error: {e}"
                }
            }

        results.append({
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

    # Save summary JSON
    summary_data = {
        "metadata": metadata,
        "address_consistency_audit": address_audit,
        "visual_assets_validation": results
    }
    summary_path = os.path.join(output_dir, "validation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    # Save HTML Report
    html_report_path = os.path.join(output_dir, "validation_report.html")
    generate_html_report(metadata, results, html_report_path, address_audit)

    print(f"\n=======================================================")
    print(f"Validation Completed for {os.path.basename(pdf_path)}")
    print(f"Summary JSON: {summary_path}")
    print(f"HTML Report:  {html_report_path}")
    print(f"=======================================================\n")

    return html_report_path, summary_path

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\Downloads\16 Firwood Dr.pdf"
    process_appraisal_file(target)
