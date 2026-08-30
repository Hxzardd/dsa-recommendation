import json
import os
import random
import hashlib
from collections import defaultdict

random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(BASE, "data")

POOL_TYPES = ["needs_attention", "weak_topic", "current_topic"]
POOL_PRIORITY = {"needs_attention": 0.06, "weak_topic": 0.03, "current_topic": 0.0}

def stable_difficulty(problem_id):
    h = hashlib.sha256(problem_id.encode()).hexdigest()
    return round(0.1 + 0.8 * (int(h[:8], 16) / 0xFFFFFFFF), 3)

def load_problem_pool(real_difficulty_map=None):
    with open(os.path.join(UPLOADS, "problem_topic_edges_normalized.json")) as f:
        pt_edges = json.load(f)
    problem_topics = defaultdict(list)
    for edge in pt_edges:
        problem_topics[edge["source"]].append(edge["target"])
    with open(os.path.join(UPLOADS, "1000_manifest_final.json")) as f:
        manifest = json.load(f)
    problems = []
    for p in manifest:
        slug = p["title_slug"]
        topics = problem_topics.get(slug, [])
        if not topics:
            continue
        if real_difficulty_map and slug in real_difficulty_map:
            difficulty = round(float(real_difficulty_map[slug]), 3)
        else:
            difficulty = stable_difficulty(p["problem_id"])
        likes = p.get("likes", 0) or 0
        dislikes = p.get("dislikes", 0) or 0
        acceptance = round(likes / (likes + dislikes), 3) if likes + dislikes > 0 else 0.5
        problems.append({"problem_id": p["problem_id"], "title_slug": slug,
                         "topics": topics, "difficulty_score": difficulty,
                         "acceptance_rate": acceptance})
    return problems

def make_user_profile(all_topics):
    archetype = random.choice(["beginner", "intermediate", "advanced", "rusty"])
    if archetype == "beginner":
        n, mr, rr = random.randint(3,12), (0.05,0.4), (0,10)
    elif archetype == "intermediate":
        n, mr, rr = random.randint(10,30), (0.3,0.7), (0,25)
    elif archetype == "advanced":
        n, mr, rr = random.randint(25,60), (0.6,0.95), (0,40)
    else:
        n, mr, rr = random.randint(15,40), (0.4,0.85), (20,90)
    touched = random.sample(all_topics, min(n, len(all_topics)))
    profile = {}
    for t in touched:
        mastery = round(random.uniform(*mr), 3)
        profile[t] = {"mastery": mastery,
                      "half_life": round(1.0 + mastery*random.uniform(5,25), 3),
                      "days_since": round(random.uniform(*rr), 2)}
    return archetype, profile

def p_recall(half_life, days_since):
    return 2 ** (-days_since / half_life) if half_life > 0 else 0.0

def compute_features(profile, problem, recent_topics):
    topics = problem["topics"]
    masteries = [profile.get(t, {}).get("mastery", 0.15) for t in topics]
    bkt_avg = sum(masteries) / max(1, len(masteries))
    recalls, urg, hls, days = [], [], [], []
    for t in topics:
        st = profile.get(t)
        if st:
            pr = p_recall(st["half_life"], st["days_since"]); hls.append(st["half_life"]); days.append(st["days_since"])
        else:
            pr = 0.5; hls.append(1.0); days.append(0.0)
        recalls.append(pr); urg.append(1.0 - pr)
    overlap = sum(1 for t in topics if t in profile)
    base_sim = overlap / max(1, len(topics))
    sim = round(min(1.0, max(0.0, base_sim + random.uniform(-0.15, 0.15))), 4)
    recent_hits = sum(1 for t in topics if t in recent_topics)
    variety = round(max(0.0, 1.0 - recent_hits / max(1, len(topics))), 4)
    return {"bkt_mastery_avg": round(bkt_avg,4),
            "hlr_urgency_avg": round(sum(urg)/max(1,len(urg)),4),
            "p_recall_avg": round(sum(recalls)/max(1,len(recalls)),4),
            "half_life_avg": round(sum(hls)/max(1,len(hls)),4),
            "days_since_last_review": round(sum(days)/max(1,len(days)),4),
            "difficulty_score": problem["difficulty_score"],
            "acceptance_rate": problem["acceptance_rate"],
            "similarity_score": sim, "variety_score": variety, "topic_overlap": overlap}

def prerequisites_met(profile, problem, prereq_table, threshold=0.5):
    required = set()
    for t in problem["topics"]:
        required.update(prereq_table.get(t, []))
    if not required:
        return 1.0
    met = sum(1 for r in required if profile.get(r, {}).get("mastery", 0.0) >= threshold)
    return met / len(required)

def rule_relevance(f, pool, prereq_frac):
    m, u, rec, d, sim, var = (f["bkt_mastery_avg"], f["hlr_urgency_avg"], f["p_recall_avg"],
                              f["difficulty_score"], f["similarity_score"], f["variety_score"])
    diff_match = max(0.0, 1.0 - 2.0*abs(d - (m + 0.1)))
    weakness = (1.0 - m) * diff_match
    rel = 0.35*diff_match + 0.25*u + 0.25*weakness + 0.15*sim
    if pool == "needs_attention" and rec < 0.4: rel *= 1.30
    elif pool == "weak_topic" and m < 0.3: rel *= 1.20
    elif pool == "current_topic" and u < 0.5: rel *= 1.10
    rel *= (0.7 + 0.3*var)
    rel *= (0.3 + 0.7*prereq_frac)
    if m < 0.3 and d > 0.7: rel *= 0.4
    if m > 0.75 and d < 0.3: rel *= 0.5
    rel += POOL_PRIORITY[pool]
    return round(min(1.0, max(0.0, rel)), 4)

def build_prereq_table(jac=0.1):
    with open(os.path.join(UPLOADS, "topic_topic_edges_normalized.json"), encoding="utf-8-sig") as f:
        tt = json.load(f)
    pr = defaultdict(list)
    for e in tt:
        if e.get("jaccard", 0) > jac:
            pr[e["target"]].append(e["source"])
    return pr

def main(n_users=800, candidates_per_user=25, out_path=None, real_difficulty_map=None):
    if real_difficulty_map is None:
        mp = os.path.join(BASE, "real_difficulty_map.json")
        if os.path.exists(mp):
            real_difficulty_map = json.load(open(mp))
            print(f"Loaded real difficulty for {len(real_difficulty_map)} problems")
        else:
            print("No real_difficulty_map.json - using PLACEHOLDER difficulty. Run export_difficulty.py first.")
    problems = load_problem_pool(real_difficulty_map)
    all_topics = sorted({t for p in problems for t in p["topics"]})
    prereq_table = build_prereq_table()
    topic_to_problems = defaultdict(list)
    for p in problems:
        for t in p["topics"]:
            topic_to_problems[t].append(p)
    rows = []
    for _ in range(n_users):
        _, profile = make_user_profile(all_topics)
        seen = list(profile.keys())
        recent = set(random.sample(seen, min(5, len(seen)))) if seen else set()
        n_rel = int(candidates_per_user*0.8); n_exp = candidates_per_user - n_rel
        rel_pool = []
        for t in seen: rel_pool.extend(topic_to_problems.get(t, []))
        cands = []
        if rel_pool: cands.extend(random.sample(rel_pool, min(n_rel, len(rel_pool))))
        cands.extend(random.sample(problems, min(n_exp, len(problems))))
        seen_ids, uniq = set(), []
        for c in cands:
            if c["problem_id"] not in seen_ids:
                seen_ids.add(c["problem_id"]); uniq.append(c)
        for prob in uniq:
            pool = random.choice(POOL_TYPES)
            feats = compute_features(profile, prob, recent)
            pf = prerequisites_met(profile, prob, prereq_table)
            row = dict(feats)
            row["pool_needs_attention"] = 1 if pool=="needs_attention" else 0
            row["pool_weak_topic"] = 1 if pool=="weak_topic" else 0
            row["pool_current_topic"] = 1 if pool=="current_topic" else 0
            row["relevance"] = rule_relevance(feats, pool, pf)
            rows.append(row)
    seen_keys, dedup = set(), []
    for r in rows:
        k = tuple(round(r[x],4) for x in sorted(r))
        if k in seen_keys: continue
        seen_keys.add(k); dedup.append(r)
    rows = dedup
    import bisect
    rels = sorted(r["relevance"] for r in rows); n = len(rels)
    for r in rows:
        r["relevance"] = round(bisect.bisect_left(rels, r["relevance"]) / max(1, n-1), 4)
    if out_path is None:
        out_path = os.path.join(BASE, "lightgbm_dataset.jsonl")
    with open(out_path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"Generated {len(rows)} rows. Written to {out_path}")

if __name__ == "__main__":
    main()
