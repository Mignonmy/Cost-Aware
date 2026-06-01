import json
import random
from collections import Counter

# ===========================
STATS_PATH = "corpus/data/domain_stats.json"
TIER_MAP_PATH = "corpus/data/domain_tier_map_newtiertable.json"
OUTPUT_NEW_TIER_MAP = "corpus/data/domain_tier_map_newtiertable_final_balanced.json"
TOP_K = 20000
# ===========================

random.seed(42)

# ---------- Load ----------
with open(STATS_PATH, "r", encoding="utf-8") as f:
    stats = Counter(json.load(f))

with open(TIER_MAP_PATH, "r", encoding="utf-8") as f:
    tier_map = json.load(f)

sorted_domains = stats.most_common()
total_segments = sum(stats.values())

top_domains = [d[0] for d in sorted_domains[:TOP_K]]
longtail_domains = [d[0] for d in sorted_domains[TOP_K:]]

print("="*60)
print(f"Total corpus segments: {total_segments:,}")
print(f"Top {TOP_K} domains: {len(top_domains)}")
print(f"Longtail domains: {len(longtail_domains)}")
print("="*60)
print()



print("Top20k GPT Classification Results")
print("-"*60)

# Domain-level distribution
domain_tier_counter = Counter()
for d in top_domains:
    tier = tier_map.get(d, 3)
    domain_tier_counter[tier] += 1

print("Domain-level distribution:")
for tier in sorted(domain_tier_counter):
    p = domain_tier_counter[tier] / TOP_K
    print(f"Tier {tier}: {domain_tier_counter[tier]:,} domains ({p:.2%})")

print()

# Segment-level distribution
top_segment_counter = Counter()
top_total_segments = 0

for d in top_domains:
    tier = tier_map.get(d, 3)
    seg = stats[d]
    top_segment_counter[tier] += seg
    top_total_segments += seg

print("Segment-level distribution (within Top20k):")
tier_proportion = {}

for tier in sorted(top_segment_counter):
    p = top_segment_counter[tier] / top_total_segments
    tier_proportion[tier] = p
    print(f"Tier {tier}: {top_segment_counter[tier]:,} segments ({p:.2%})")

coverage_ratio = top_total_segments / total_segments

print()
print(f"Top20k covers {top_total_segments:,} segments "
      f"({coverage_ratio:.2%} of entire corpus)")
print()
print("="*60)
print()

# =====================================================
# 1️⃣ Longtail total segments
# =====================================================

longtail_total_segments = sum(stats[d] for d in longtail_domains)
print(f"Longtail total segments: {longtail_total_segments:,}")
print()

# =====================================================
# 2️⃣ Compute target segments for longtail
# =====================================================

target_longtail_segments = {
    tier: int(tier_proportion[tier] * longtail_total_segments)
    for tier in tier_proportion
}

print("Target longtail allocation:")
for tier in sorted(target_longtail_segments):
    print(f"Tier {tier}: {target_longtail_segments[tier]:,}")
print()

# =====================================================
# 3️⃣ Randomly assign longtail
# =====================================================

random.shuffle(longtail_domains)

new_tier_map = tier_map.copy()
assigned_longtail_segments = Counter()

tier_list = sorted(target_longtail_segments.keys())
tier_pointer = 0

for d in longtail_domains:
    seg = stats[d]

    while (
        tier_pointer < len(tier_list)
        and assigned_longtail_segments[tier_list[tier_pointer]]
        >= target_longtail_segments[tier_list[tier_pointer]]
    ):
        tier_pointer += 1

    if tier_pointer >= len(tier_list):
        tier = tier_list[-1]
    else:
        tier = tier_list[tier_pointer]

    new_tier_map[d] = tier
    assigned_longtail_segments[tier] += seg

print("Actual longtail assigned:")
for tier in sorted(assigned_longtail_segments):
    print(f"Tier {tier}: {assigned_longtail_segments[tier]:,}")
print()

# =====================================================
# 4️⃣ Final corpus distribution
# =====================================================

final_segment_counter = Counter()

for domain, seg in stats.items():
    tier = new_tier_map.get(domain, 3)
    final_segment_counter[tier] += seg

print("="*60)
print("Final corpus segment-level distribution")
print("-"*60)

for tier in sorted(final_segment_counter):
    p = final_segment_counter[tier] / total_segments
    print(f"Tier {tier}: {final_segment_counter[tier]:,} segments ({p:.2%})")

print("="*60)
print()

# =====================================================
# 5️⃣ Save
# =====================================================

with open(OUTPUT_NEW_TIER_MAP, "w", encoding="utf-8") as f:
    json.dump(new_tier_map, f)

print("Final tier map saved.")