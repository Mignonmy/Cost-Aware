import json
from collections import Counter

# ===========================
# CONFIG
# ===========================
TIER_MAP_PATH = "corpus/data/domain_tier_map_newtiertable_final_balanced.json"
OUTPUT_MERGED_TIER_MAP = "corpus/data/domain_tier_map_newtiertable_final_balanced_merge23tiers.json"
# ===========================

# ---------- Load tier map ----------
with open(TIER_MAP_PATH, "r", encoding="utf-8") as f:
    tier_map = json.load(f)

print(f"Loaded {len(tier_map)} domains from tier map")
print()

# ---------- Merge tier2 and tier3 into new tier2 ----------
merged_tier_map = {}
tier_counter = Counter()

for domain, tier in tier_map.items():
    # Merge tier2 and tier3 into new tier2
    if tier in [2, 3]:
        new_tier = 2
    else:
        new_tier = tier
    
    merged_tier_map[domain] = new_tier
    tier_counter[new_tier] += 1

# ---------- Statistics ----------
print("="*60)
print("Final Tier Distribution (Domain-level)")
print("="*60)

total_domains = len(merged_tier_map)

for tier in sorted(tier_counter.keys()):
    count = tier_counter[tier]
    percentage = (count / total_domains) * 100
    print(f"Tier {tier}: {count:,} domains ({percentage:.2f}%)")

print("-"*60)
print(f"Total: {total_domains:,} domains")
print("="*60)
print()

# ---------- Save merged tier map ----------
with open(OUTPUT_MERGED_TIER_MAP, "w", encoding="utf-8") as f:
    json.dump(merged_tier_map, f, indent=2)

print(f"Merged tier map saved to: {OUTPUT_MERGED_TIER_MAP}")
