"""Build the hand-labelled held-out validation set for the fallback classifiers.

Every experience entry in the 25 real profiles, labelled by me by hand from the
role title, the company, the description and `skills_used`. This set is
**never** trained on -- it exists solely to measure the distilled LR and the
MiniLM centroid on the task they actually perform.

Provenance, stated plainly because it bounds what the numbers mean:

* The labels are my judgement, produced after I had written the lexicon and
  seen its output, but before either fallback classifier existed. So the set is
  independent of the models being measured, and **not** independent of me. At
  real scale the correct version of this set is recruiter-graded.
* Entries a human genuinely cannot call from the row alone are labelled `null`
  and excluded from accuracy, with the count reported. Forcing a label onto
  "Software Intern" with an empty description would measure my guessing, not
  the classifier's.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (candidate_id, entry_index) -> role_family, or None where a human cannot tell.
HAND_LABELS: dict[tuple[str, int], str | None] = {
    ("SDB_10001", 0): "backend",          # Senior SWE @ Razorpay, payments ledger, 12k rps
    ("SDB_10001", 1): "backend",          # SWE @ Freshworks, billing APIs, Django/FastAPI
    ("SDB_10002", 0): "ml_engineer",      # MLE @ Flipkart, cross-encoder reranker
    ("SDB_10002", 1): "data_scientist",   # Data Scientist @ Mu Sigma, client analytics
    ("SDB_10003", 0): "data_engineer",    # Data Engineer II @ Zeta, 300 Airflow DAGs
    ("SDB_10003", 1): "data_engineer",    # Data Engineer @ Persistent
    ("SDB_10004", 0): "data_analyst",     # Data Analyst @ Deloitte, dashboards + SQL
    ("SDB_10005", 0): "non_engineering",  # Founder & CEO @ Stealth
    ("SDB_10005", 1): "non_engineering",  # Program Manager @ Amazon
    ("SDB_10006", 0): "backend",          # Backend Developer @ TCS, Node services
    ("SDB_10007", 0): "ml_engineer",      # Applied Scientist @ Sprinklr, RAG eval, 7B finetune
    ("SDB_10007", 1): "ml_engineer",      # Research Assistant @ IISc, information retrieval lab
    ("SDB_10008", 0): "qa",               # QA Engineer @ Infostretch, Selenium
    ("SDB_10009", 0): "devops_sre",       # Staff Engineer @ PhonePe, internal developer platform
    ("SDB_10009", 1): "devops_sre",       # SRE @ Grofers
    ("SDB_10010", 0): "data_scientist",   # Data Science Intern @ Cognifront
    ("SDB_10011", 0): "data_engineer",    # Senior Data Engineer @ Publicis Sapient, lakehouse
    ("SDB_10011", 1): "data_engineer",    # Data Engineer @ Xebia
    ("SDB_10012", 0): "frontend",         # Frontend Engineer @ CRED, LCP work
    ("SDB_10012", 1): "frontend",         # UI Developer @ Zeta, React
    ("SDB_10013", 0): "backend",          # SDE II @ Salesforce, Java/Spring Boot
    ("SDB_10013", 1): "backend",          # SDE I @ Infosys, Java
    ("SDB_10014", 0): "ml_engineer",      # AI Engineer @ FutureTech, text-to-SQL agent, RAG
    ("SDB_10014", 1): None,               # "Software Intern" @ Casepoint: no description, no skills
    ("SDB_10015", 0): "ml_engineer",      # ML Engineer @ Swiggy, home feed ranking, Triton
    ("SDB_10015", 1): "ml_engineer",      # ML Engineer @ Myntra, TensorFlow
    ("SDB_10016", 0): "data_analyst",     # Product Analyst @ Meesho, 40+ A/B tests
    ("SDB_10016", 1): "data_analyst",     # Business Analyst @ ZS Associates, Excel
    ("SDB_10017", 0): "fullstack",        # Full Stack Developer @ Growisto, MERN
    ("SDB_10017", 1): "backend",          # SWE @ Yellow.ai, Node.js
    ("SDB_10017", 2): None,               # "Associate Engineer" @ Impetus: JavaScript only, could be either
    ("SDB_10018", 0): "data_engineer",    # Analytics Engineer @ Chargebee, dbt/Snowflake, 340 models
    ("SDB_10018", 1): "data_analyst",     # Data Analyst @ Zoho
    ("SDB_10019", 0): "non_engineering",  # Design Engineer @ Hero MotoCorp, AutoCAD (mechanical)
    ("SDB_10020", 0): "engineering_manager",  # EM @ Atlassian
    ("SDB_10020", 1): "ml_engineer",      # Senior ML Engineer @ Uber
    ("SDB_10020", 2): None,               # "Software Engineer" @ Yahoo 2013: no description, no skills
    ("SDB_10021", 0): "data_engineer",    # Freelance Data Engineer, BigQuery + Dataflow
    ("SDB_10021", 1): "data_engineer",    # Data Engineer @ Dunzo
    ("SDB_10022", 0): "backend",          # SDE III @ Amazon, serverless event pipelines
    ("SDB_10022", 1): "backend",          # SDE II @ Adobe, Python
    ("SDB_10023", 0): "non_engineering",  # HR Executive @ Zydus
    ("SDB_10024", 0): "ml_engineer",      # ML Engineer @ Ather, on-vehicle perception, INT8
    ("SDB_10024", 1): "ml_engineer",      # Computer Vision Intern @ Qualcomm
    ("SDB_10025", 0): "backend",          # Backend Engineer @ Paytm, wallet + refunds
}


def main() -> None:
    profiles = [
        json.loads(line)
        for line in (ROOT / "data" / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = []
    for profile in profiles:
        for index, entry in enumerate(profile.get("experience") or []):
            key = (profile["id"], index)
            if key not in HAND_LABELS:
                raise SystemExit(f"unlabelled entry: {key} -> {entry.get('role')}")
            rows.append(
                {
                    "candidate_id": profile["id"],
                    "entry_index": index,
                    "title": entry.get("role"),
                    "company": entry.get("company_name"),
                    "context": " ".join(
                        filter(
                            None,
                            [
                                entry.get("description") or "",
                                " ".join(entry.get("skills_used") or []),
                            ],
                        )
                    ).strip(),
                    "role_family": HAND_LABELS[key],
                }
            )

    out = ROOT / "config" / "title_labels_holdout.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    labelled = sum(1 for r in rows if r["role_family"])
    print(f"wrote {len(rows)} entries to {out.relative_to(ROOT)}")
    print(f"  {labelled} labelled, {len(rows) - labelled} deliberately unlabelled")


if __name__ == "__main__":
    main()
