# use GPT to judgement the top 20,000 domains
import json
import time
from collections import Counter
from openai import OpenAI

# ===========================
# CONFIG
# ===========================
STATS_PATH = "corpus/data/domain_stats.json"
OUTPUT_TIER_MAP = "corpus/data/domain_tier_map_newtiertable.json"

TOP_K = 20000
BATCH_SIZE = 1000
MODEL = "gpt-4.1-mini"

from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
# ===========================


# ---------- Load domain stats ----------
with open(STATS_PATH, "r", encoding="utf-8") as f:
    stats = Counter(json.load(f))

total_segments = sum(stats.values())
sorted_domains = stats.most_common()

top_domains = [d[0] for d in sorted_domains[:TOP_K]]
long_tail_domains = [d[0] for d in sorted_domains[TOP_K:]]

print(f"Top domains for GPT classification: {len(top_domains)}")
print(f"Temp: Long tail domains auto-assign to Tier 3: {len(long_tail_domains)}")


# ---------- GPT Prompt Template ----------
SYSTEM_PROMPT = """
You are classifying website domains into access-friction tiers.

Tier definitions:

Tier 0:
Open community resources (Wikipedia, public forums, UGC platforms).

Tier 1:
General open web content (blogs, tutorials, news sites).

Tier 2:
Curated/official/professional sources with moderate friction (official vendor docs & API refs, support KBs, standards/gov technical docs; may include free-login or soft gating)

Tier 3:
High-friction restricted sources (paid paywalls, enterprise SSO portals, proprietary aggregators)

Return ONLY a JSON object mapping domain to tier integer.
Do not explain.
"""

def safe_json_parse(content):
    content = content.strip()

    if content.startswith("```"):
        content = content.split("```")[1]

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        content = content[start:end+1]

    if not content.startswith("{"):
        content = "{" + content

    try:
        return json.loads(content)
    except Exception as e:
        print("Still JSON error:", e)
        print("Raw content preview:")
        print(content[:500])
        return {}
    
def classify_batch(domains):
    domain_list_str = "\n".join(domains)

    user_prompt = f"""
Classify the following domains:

{domain_list_str}
"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content.strip()

    try:
        # return json.loads(content)
        return safe_json_parse(content)
    except:
        print("JSON parse error. Raw output:")
        print(content)
        return {}


# ---------- Run GPT in batches ----------
tier_map = {}

for i in range(0, len(top_domains), BATCH_SIZE):
    batch = top_domains[i:i+BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE + 1}...")

    result = classify_batch(batch)
    tier_map.update(result)

    time.sleep(1)

# ---------- Noted: this is temporary ----------
# ---------- Assign long tail to Tier 3 ----------
for d in long_tail_domains:
    tier_map[d] = 3


# ---------- Save mapping ----------
with open(OUTPUT_TIER_MAP, "w", encoding="utf-8") as f:
    json.dump(tier_map, f)

print("Tier mapping saved.")


# ---------- Compute Tier Distribution ----------
tier_segment_counter = Counter()

for domain, count in stats.items():
    tier = tier_map.get(domain, 3)
    tier_segment_counter[tier] += count

print("\nTier Distribution (by segments):")

for tier in sorted(tier_segment_counter):
    proportion = tier_segment_counter[tier] / total_segments
    print(f"Tier {tier}: {tier_segment_counter[tier]:,} segments ({proportion:.2%})")
