import os
import spacy
import json
import hashlib
from datetime import datetime
from spacy_layout import spaCyLayout

# Directory containing your PDF abstracts
abstracts_dir = os.path.expanduser("~/abstracts")

# Enable GPU and load the scientific spaCy model
try:
    spacy.require_gpu()
    print("spaCy GPU enabled.")
except Exception as e:
    print(f"spaCy GPU not enabled: {e}\nProceeding with CPU.")
nlp = spacy.load("en_core_sci_scibert")
layout = spaCyLayout(nlp)

# Output files
output_path = "processed_abstracts_sentences.jsonl"
tracking_path = "processed_files_tracking.json"

def get_file_hash(filepath):
    """Generate MD5 hash of file for change detection"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_tracking_data():
    """Load tracking data of processed files"""
    if os.path.exists(tracking_path):
        with open(tracking_path, "r") as f:
            return json.load(f)
    return {}

def save_tracking_data(tracking_data):
    """Save tracking data of processed files"""
    with open(tracking_path, "w") as f:
        json.dump(tracking_data, f, indent=2)

def process_pdf_file(filename, pdf_path, out_f):
    """Process a single PDF file and write sentences to output"""
    print(f"Processing {filename}...")
    doc = layout(pdf_path)
    sentence_count = 0
    
    # Each span in doc.spans["layout"] is a block/region with layout info
    for span in doc.spans.get("layout", []):
        text = span.text.strip()
        if text:
            # Use spaCy parser-based sentence segmentation on the span
            span_doc = nlp(text)
            for sent in span_doc.sents:
                sent_text = sent.text.strip()
                if sent_text:
                    record = {
                        "filename": filename,
                        "page": getattr(span, "page_num", None),
                        "block_label": span.label_,
                        "sentence": sent_text,
                        "entities": [(ent.text, ent.label_) for ent in sent.ents]
                    }
                    out_f.write(json.dumps(record) + "\n")
                    sentence_count += 1
    
    print(f"  → Extracted {sentence_count} sentences from {filename}")
    return sentence_count

def main():
    """Main processing function with duplicate prevention"""
    # Load tracking data
    tracking_data = load_tracking_data()
    
    # Get all PDF files in directory
    pdf_files = [f for f in os.listdir(abstracts_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found in abstracts directory")
        return
    
    # Check which files need processing
    files_to_process = []
    skipped_files = []
    
    for filename in pdf_files:
        pdf_path = os.path.join(abstracts_dir, filename)
        file_hash = get_file_hash(pdf_path)
        file_stat = os.stat(pdf_path)
        
        # Check if file needs processing
        if filename in tracking_data:
            stored_hash = tracking_data[filename].get("hash")
            if stored_hash == file_hash:
                print(f"⏭️  Skipping {filename} (already processed, no changes)")
                skipped_files.append(filename)
                continue
            else:
                print(f"🔄 File {filename} has changed, will reprocess")
        else:
            print(f"🆕 New file found: {filename}")
        
        files_to_process.append({
            "filename": filename,
            "path": pdf_path,
            "hash": file_hash,
            "size": file_stat.st_size,
            "modified": file_stat.st_mtime
        })
    
    if not files_to_process:
        print("✅ All files already processed and up to date!")
        return
    
    print(f"\n📊 Processing Summary:")
    print(f"  • Files to process: {len(files_to_process)}")
    print(f"  • Files skipped: {len(skipped_files)}")
    print(f"  • Total files: {len(pdf_files)}")
    
    # Decide whether to append or overwrite based on existing data
    mode = "a" if skipped_files and os.path.exists(output_path) else "w"
    
    if mode == "w":
        print(f"\n🔄 Creating new output file: {output_path}")
    else:
        print(f"\n➕ Appending to existing output file: {output_path}")
    
    # Process files
    total_sentences = 0
    with open(output_path, mode, encoding="utf-8") as out_f:
        for file_info in files_to_process:
            sentence_count = process_pdf_file(
                file_info["filename"], 
                file_info["path"], 
                out_f
            )
            total_sentences += sentence_count
            
            # Update tracking data
            tracking_data[file_info["filename"]] = {
                "hash": file_info["hash"],
                "size": file_info["size"],
                "modified": file_info["modified"],
                "processed_date": datetime.now().isoformat(),
                "sentence_count": sentence_count
            }
    
    # Save updated tracking data
    save_tracking_data(tracking_data)
    
    print(f"\n✅ Processing complete!")
    print(f"  • Processed {len(files_to_process)} files")
    print(f"  • Extracted {total_sentences} new sentences")
    print(f"  • Results {'appended to' if mode == 'a' else 'saved to'} {output_path}")
    print(f"  • Tracking data saved to {tracking_path}")

if __name__ == "__main__":
    main()
