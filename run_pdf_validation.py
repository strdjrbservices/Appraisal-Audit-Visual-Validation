import os
import sys
import json
import pymupdf
from pipeline import process_image
from appraisal_report_generator import process_appraisal_file

def main():
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\Downloads\90 She Oak Dr SW.pdf"
    if not os.path.exists(pdf_file):
        # Fallback to search Downloads folder
        downloads_dir = r"C:\Users\Admin\Downloads"
        candidates = [f for f in os.listdir(downloads_dir) if f.lower().endswith(".pdf")]
        if candidates:
            pdf_file = os.path.join(downloads_dir, candidates[0])
            print(f"Target PDF not specified, using: {pdf_file}")
        else:
            print("No appraisal PDF file found.")
            return

    html_path, summary_path = process_appraisal_file(pdf_file)
    print(f"\nAll validations complete!")
    print(f"HTML Report: {html_path}")
    print(f"Summary JSON: {summary_path}")

if __name__ == "__main__":
    main()
