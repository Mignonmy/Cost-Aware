# extract the domain from the URL field and calculate the frequency
import os
import json
from urllib.parse import urlparse
from collections import Counter
import matplotlib.pyplot as plt


DATA_DIR = "corpus/data/msmarco_v2.1_doc_segmented"
OUTPUT_STATS = "corpus/domain_stats.json"
TOP_N = 30
PLOT = False   


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


def compute_domain_stats():
    counter = Counter()
    files = collect_json_files(DATA_DIR)

    print(f"Found {len(files)} files.\n")

    total_lines = 0

    for file_idx, filepath in enumerate(files):
        print(f"[{file_idx+1}/{len(files)}] Processing {os.path.basename(filepath)}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                record = json.loads(line)
                url = record.get("url", "")
                domain = extract_domain(url)
                if domain:
                    counter[domain] += 1

                total_lines += 1

                if total_lines % 100000 == 0:
                    print(f"  Processed {total_lines:,} segments...")

    print("\nFinished.")
    print(f"Total segments processed: {total_lines:,}")
    print(f"Total unique domains: {len(counter):,}")


    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(counter, f)

    print(f"Domain stats saved to: {OUTPUT_STATS}")

    return counter


def load_stats():
    with open(OUTPUT_STATS, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Counter(data)


def plot_top_domains(counter):
    most_common = counter.most_common(TOP_N)
    domains = [d[0] for d in most_common]
    counts = [d[1] for d in most_common]

    plt.figure()
    plt.bar(domains, counts)
    plt.xticks(rotation=90)
    plt.xlabel("Domain")
    plt.ylabel("Frequency")
    plt.title(f"Top {TOP_N} Domains")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    if not os.path.exists(OUTPUT_STATS):
        stats = compute_domain_stats()
    else:
        print("Loading existing stats...")
        stats = load_stats()

    print("\nTop 10 domains:")
    for d, c in stats.most_common(10):
        print(f"{d:40} {c}")

    if PLOT:
        plot_top_domains(stats)