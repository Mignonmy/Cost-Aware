import json

# ===========================
# CONFIG
# ===========================
DOCID_DOMAIN_PATH = "corpus/meta_data/docid_domain_mapping.json"
DOMAIN_TIER_PATH = "corpus/data/domain_tier_map_newtiertable_final_balanced_merge23tiers.json"
OUTPUT_DOCID_TIER_PATH = "corpus/meta_data/docid_tier_mapping.json"
# ===========================

def print_sample_mapping(mapping, sample_size=5):
    """Print sample mappings"""
    print(f"\nSample mappings (first {sample_size} docids):")
    for i, (docid, tier) in enumerate(mapping.items()):
        if i >= sample_size:
            break
        print(f"{docid} → {tier}")

# ---------- Load mappings ----------
print("Loading docid->domain mapping...")
with open(DOCID_DOMAIN_PATH, "r", encoding="utf-8") as f:
    docid_to_domain = json.load(f)

print(f"Loaded {len(docid_to_domain):,} docid->domain mappings")

print("Loading domain->tier mapping...")
with open(DOMAIN_TIER_PATH, "r", encoding="utf-8") as f:
    domain_to_tier = json.load(f)

print(f"Loaded {len(domain_to_tier):,} domain->tier mappings")
print()

# ---------- Generate docid->tier mapping ----------
docid_to_tier = {}
missing_domain_count = 0

for docid, domain in docid_to_domain.items():
    tier = domain_to_tier.get(domain, None)
    
    if tier is None:
        missing_domain_count += 1
        tier = 2  # Default to tier 2 if domain not found
    
    docid_to_tier[docid] = tier

print(f"Generated {len(docid_to_tier):,} docid->tier mappings")
if missing_domain_count > 0:
    print(f"Warning: {missing_domain_count} docids had domains not in tier map (defaulted to tier 2)")
print()

# ---------- Statistics ----------
from collections import Counter

tier_counter = Counter(docid_to_tier.values())

print("="*60)
print("Docid Distribution by Tier")
print("="*60)

total_docids = len(docid_to_tier)

for tier in sorted(tier_counter.keys()):
    count = tier_counter[tier]
    percentage = (count / total_docids) * 100
    print(f"Tier {tier}: {count:,} docids ({percentage:.2f}%)")

print("-"*60)
print(f"Total: {total_docids:,} docids")
print("="*60)
print()

# ---------- Save docid->tier mapping ----------
with open(OUTPUT_DOCID_TIER_PATH, "w", encoding="utf-8") as f:
    json.dump(docid_to_tier, f, indent=2)

print(f"Docid->tier mapping saved to: {OUTPUT_DOCID_TIER_PATH}")
print_sample_mapping(docid_to_tier, sample_size=5)
