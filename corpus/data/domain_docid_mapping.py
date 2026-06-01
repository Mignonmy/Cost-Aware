# Generate docid to domain mapping
import os
import json
from urllib.parse import urlparse


DATA_DIR = "corpus/data/msmarco_v2.1_doc_segmented"
OUTPUT_MAPPING = "corpus/meta_data/docid_domain_mapping.json"



def extract_domain(url):
    if not url:
        return None
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def collect_json_files(data_dir):
    files = []
    for f in os.listdir(data_dir):
        if f.endswith(".json"):
            files.append(os.path.join(data_dir, f))
    files.sort()
    return files


def generate_docid_domain_mapping():
    """Generate mapping from docid to domain"""
    docid_to_domain = {}
    files = collect_json_files(DATA_DIR)

    print(f"Found {len(files)} files.\n")

    total_lines = 0

    for file_idx, filepath in enumerate(files):
        print(f"[{file_idx+1}/{len(files)}] Processing {os.path.basename(filepath)}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                record = json.loads(line)
                url = record.get("url", "")
                docid = record.get("docid", "")
                
                domain = extract_domain(url)
                if domain and docid:
                    docid_to_domain[docid] = domain

                total_lines += 1

                if total_lines % 100000 == 0:
                    print(f"  Processed {total_lines:,} segments...")

    print("\nFinished.")
    print(f"Total segments processed: {total_lines:,}")
    print(f"Total unique docids: {len(docid_to_domain):,}")

    with open(OUTPUT_MAPPING, "w", encoding="utf-8") as f:
        json.dump(docid_to_domain, f, indent=2)

    print(f"Docid-domain mapping saved to: {OUTPUT_MAPPING}")

    return docid_to_domain


def load_mapping():
    """Load existing mapping"""
    with open(OUTPUT_MAPPING, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def print_sample_mapping(mapping, sample_size=5):
    """Print sample mappings"""
    print(f"\nSample mappings (first {sample_size} docids):")
    for i, (docid, domain) in enumerate(mapping.items()):
        if i >= sample_size:
            break
        print(f"{docid} → {domain}")



if __name__ == "__main__":

    
    if not os.path.exists(OUTPUT_MAPPING):
        mapping = generate_docid_domain_mapping()
    else:
        print("Loading existing mapping...")
        mapping = load_mapping()

    print_sample_mapping(mapping, sample_size=5)
