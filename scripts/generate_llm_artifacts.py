"""Regenerate the offline LLM artifacts. NEVER on the default path.

`make all` uses the **committed** artifacts in `config/`, so a clean clone with
no network, no GPU and no model download reproduces every number in `out/`.
This script exists so the provenance of those artifacts is auditable and so they
can be rebuilt when the taxonomy changes.

Why a synthetic corpus exists at all
------------------------------------
Training a title classifier on the 25 real candidates gives 45 experience
entries across 12 classes -- under four examples per class, several classes with
one. That is unfittable, and any accuracy number from it would be noise. The
synthetic corpus is what makes distillation honest rather than a fig leaf: the
LLM's knowledge about what job titles mean is transferred into a ~100KB sparse
model, and the 45 real entries stay held out as a genuine validation set.

Three generation backends
-------------------------
`--backend ollama --model qwen2.5:3b-instruct` (**recommended**)
    A local Ollama model at temperature 0 with a fixed seed and a JSON schema
    constraining the output to an array of strings. This is the plan's original
    design and the one a reviewer can reproduce without me: `ollama pull
    qwen2.5:3b-instruct && make regenerate-llm-artifacts`.

`--backend claude` (how the currently committed artifact was produced)
    Written by Claude Code, the same assistant used throughout this build,
    prompted per role family. Kept as the default only because it is what the
    committed file actually came from and mislabelling provenance would be worse
    than an awkward default. See AI_LOG.md AI-005.

`--backend local --model <hf-id>`
    A local HuggingFace `transformers` model, kept for environments with neither
    Ollama nor network. Slow on CPU.

Either way the output is deterministic given its inputs and is committed, and
`make all` never regenerates it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "config" / "synthetic_titles.jsonl"

FAMILY_PROMPT = """You are building a training corpus for a job-title classifier
used on Indian tech candidate profiles.

List {n} realistic job titles that belong to the role family "{family}".
Requirements:
- Cover junior through staff/lead seniority variants.
- Include the abbreviations and messy variants that appear on real Indian
  LinkedIn profiles (SDE-2, MTS, Associate Consultant, Analytics Engineer,
  Member of Technical Staff, Sr. Engineer, and so on) where they genuinely
  belong to this family.
- Include titles used by Indian IT services companies, not only product firms.
- Do NOT include titles that would be ambiguous between families.
Return a JSON array of strings and nothing else."""


#: Constrains the reply to a bare array of strings, so a title list cannot come
#: back wrapped in prose or markdown fences. Shape only -- the model can still
#: return a title that belongs to the wrong family, which is why the distilled
#: classifier is validated on hand-labelled real entries rather than on this.
TITLES_SCHEMA = {"type": "array", "items": {"type": "string"}, "minItems": 5}


def generate_ollama(model_id: str, families: list[str], per_family: int) -> list[dict]:
    """Generate with a local Ollama model, schema-constrained and seeded."""
    sys.path.insert(0, str(ROOT / "src"))
    from saral.adapters.llm.ollama_client import OllamaClient

    client = OllamaClient(model=model_id)
    if not client.available():
        raise SystemExit(f"ollama does not have {model_id}. Run: ollama pull {model_id}")

    rows: list[dict] = []
    pinned = client.pinned_tag()
    for family in families:
        prompt = FAMILY_PROMPT.format(n=per_family, family=family)
        result = client.generate(prompt, TITLES_SCHEMA)
        text = result.get("text", "")
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < 0:
            print(f"[warn] {family}: no JSON array in response, skipped")
            continue
        try:
            titles = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            print(f"[warn] {family}: unparseable JSON, skipped")
            continue
        kept = 0
        for title in titles:
            if isinstance(title, str) and title.strip():
                rows.append({"title": title.strip(), "role_family": family, "source": pinned})
                kept += 1
        print(f"  {family}: {kept} titles ({result.get('wall_ms', 0) / 1000:.1f}s)")
    return rows


def generate_local(model_id: str, families: list[str], per_family: int) -> list[dict]:
    """Generate with a local instruct model. Slow on CPU by design -- see INFRA.md."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()

    rows: list[dict] = []
    for family in families:
        prompt = FAMILY_PROMPT.format(n=per_family, family=family)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        out = model.generate(
            ids,
            max_new_tokens=900,
            do_sample=False,  # greedy: the artifact must be reproducible
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0][ids.shape[1] :], skip_special_tokens=True)
        start, end = text.find("["), text.rfind("]")
        if start < 0 or end < 0:
            print(f"[warn] {family}: no JSON array in response, skipped")
            continue
        try:
            titles = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            print(f"[warn] {family}: unparseable JSON, skipped")
            continue
        for title in titles:
            if isinstance(title, str) and title.strip():
                rows.append(
                    {"title": title.strip(), "role_family": family, "source": model_id}
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["claude", "ollama", "local"], default="claude"
    )
    parser.add_argument("--model", default="qwen2.5:3b-instruct")
    parser.add_argument("--per-family", type=int, default=45)
    args = parser.parse_args()

    if args.backend == "claude":
        print(
            "The committed corpus at config/synthetic_titles.jsonl was generated by\n"
            "Claude Code using the per-family prompt in this file (see AI_LOG.md AI-005).\n"
            "Re-run with --backend local --model <hf-id> to regenerate with a local model."
        )
        if CORPUS.exists():
            rows = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l.strip()]
            families = sorted({r["role_family"] for r in rows})
            print(f"current corpus: {len(rows)} titles across {len(families)} families")
        return

    sys.path.insert(0, str(ROOT / "src"))
    from saral.contracts.taxonomy import RoleFamily

    families = [f.value for f in RoleFamily]
    if args.backend == "ollama":
        rows = generate_ollama(args.model, families, args.per_family)
    else:
        rows = generate_local(args.model, families, args.per_family)

    seen: set[tuple[str, str]] = set()
    deduped = []
    for row in rows:
        key = (row["title"].casefold(), row["role_family"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)

    with CORPUS.open("w", encoding="utf-8", newline="\n") as handle:
        for row in deduped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(deduped)} titles to {CORPUS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
