import os



import os
import spacy
import json
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

# Output file to save processed sentence-level chunks
output_path = "processed_abstracts_sentences.jsonl"

with open(output_path, "w", encoding="utf-8") as out_f:
    for filename in os.listdir(abstracts_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(abstracts_dir, filename)
            print(f"Processing {filename}...")
            doc = layout(pdf_path)
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
print(f"Processing complete. Sentence-level results saved to {output_path}")
