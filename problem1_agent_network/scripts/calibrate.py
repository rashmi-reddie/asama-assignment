"""Print the full funnel scores for sample posts (helps tune thresholds / explain behaviour)."""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.embeddings import build_embedder
from app.reranker import Candidate, HeuristicReranker
from app.demo_agents import SEED_COMPANIES

POSTS = [
    ("Swiggy", "Food Delivery", "Diwali Food Delivery Discount", "Swiggy offering flat 5% off on all delivery orders above Rs 300 during Diwali week."),
    ("Bajaj Auto", "Automobile", "Zero down payment on Pulsar", "Bajaj Auto launching zero down payment and low EMI financing on Pulsar motorcycles this festive season."),
    ("Flipkart", "E-commerce", "Big Billion smartphone sale", "Flipkart festive sale with up to 40% off on smartphones and electronics, looking for bank card and EMI partners."),
    ("HDFC Bank", "Banking", "Quarterly board meeting notes", "HDFC Bank announces quarterly board meeting and dividend schedule for shareholders."),
]

async def main():
    emb = build_embedder(os.getenv("ASAMA_EMBEDDINGS", "auto")); rr = HeuristicReranker(emb.dense_mid)
    print("embedder:", emb.name)
    mats = emb.embed([f"{c['company']}. {c['industry']}. {c['manifesto']}" for c in SEED_COMPANIES])
    for author, ind, title, content in POSTS:
        v = emb.embed([f"{title}. {content}"])[0]
        cands = [Candidate(c["company"], c["company"], c["industry"], c["manifesto"], float(mats[i] @ v))
                 for i, c in enumerate(SEED_COMPANIES) if c["company"] != author]
        sc = {s.agent_id: s.score for s in await rr.score(title, content, ind, cands)}
        print(f"\n== {author}: {title}")
        for c in sorted(cands, key=lambda c: -sc[c.agent_id]):
            flag = "NOTIFY" if sc[c.agent_id] >= 0.75 else "      "
            print(f"  {flag} {c.company_name:13s} dense={c.vector_score:.2f} final={sc[c.agent_id]:.3f}")
asyncio.run(main())
