"""ContractLens — Legal Contract Clause Extraction demo on Hugging Face Spaces.

Pipeline: Retrieve → Extract (LiteLLM) → Verify (LLM-as-judge) via LangGraph.
"""

import os
import sys
import traceback
import uuid
from typing import Any

# Make the contractlens package importable from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

st.set_page_config(
    page_title="ContractLens — Legal Clause Extractor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": "https://github.com/charanyellanki/ContractLens"},
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.badge-verified {
    background: #d1fae5; color: #065f46;
    padding: 2px 10px; border-radius: 20px;
    font-size: .8rem; font-weight: 600;
}
.badge-rejected {
    background: #fee2e2; color: #7f1d1d;
    padding: 2px 10px; border-radius: 20px;
    font-size: .8rem; font-weight: 600;
}
.clause-card {
    border-left: 3px solid #2563eb;
    background: #f8fafc;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    margin: 6px 0 10px 0;
    font-size: .9rem;
    line-height: 1.55;
}
.clause-card.rejected { border-left-color: #ef4444; background: #fff5f5; }
.step-done   { color: #059669; font-weight: 700; }
.step-active { color: #2563eb; font-weight: 700; }
.step-todo   { color: #9ca3af; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sample contract ─────────────────────────────────────────────────────────────
SAMPLE_CONTRACT = """\
MUTUAL NON-DISCLOSURE AND SOFTWARE LICENSE AGREEMENT

This Agreement is entered into as of March 1, 2024, by and between Acme Technologies Inc.,
a Delaware corporation ("Licensor"), and Beta Solutions LLC, a California limited liability
company ("Licensee").

1. CONFIDENTIALITY
Each Party agrees to maintain in strict confidence all Confidential Information received from
the other Party. "Confidential Information" means any proprietary information, technical data,
trade secrets, or know-how disclosed by either Party in connection with this Agreement.
This confidentiality obligation shall survive termination of this Agreement for five (5) years.

2. LICENSE GRANT
Subject to the terms of this Agreement, Licensor hereby grants to Licensee a non-exclusive,
non-transferable, limited license to use the Software solely for Licensee's internal business
purposes. Licensee shall not sublicense, sell, resell, transfer, assign, or otherwise dispose
of the Software without Licensor's prior written consent.

3. PAYMENT TERMS
Licensee agrees to pay Licensor an annual license fee of $50,000, due within 30 days of invoice.
All payments shall be made in U.S. dollars. Late payments shall accrue interest at 1.5% per month
or the maximum rate permitted by law, whichever is less.

4. INTELLECTUAL PROPERTY
All intellectual property rights in and to the Software, including all modifications,
enhancements, and derivative works, shall remain the exclusive property of Licensor.
Licensee shall not acquire any ownership interest in the Software by virtue of this Agreement.

5. LIMITATION OF LIABILITY
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL,
OR PUNITIVE DAMAGES, REGARDLESS OF THE CAUSE OF ACTION OR THE THEORY OF LIABILITY, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. EACH PARTY'S TOTAL CUMULATIVE LIABILITY ARISING
OUT OF OR RELATED TO THIS AGREEMENT SHALL NOT EXCEED THE FEES PAID BY LICENSEE IN THE
TWELVE (12) MONTHS PRECEDING THE CLAIM.

6. INDEMNIFICATION
Licensor shall indemnify, defend, and hold harmless Licensee and its officers, directors, and
employees from and against any third-party claims, damages, or expenses (including reasonable
attorneys' fees) arising out of or relating to Licensor's infringement of any patent, copyright,
or trade secret rights. Licensee shall promptly notify Licensor of any such claim and cooperate
in the defense thereof.

7. TERMINATION
Either Party may terminate this Agreement upon thirty (30) days written notice to the other Party.
Licensor may terminate this Agreement immediately upon Licensee's material breach of any provision
hereof. Upon termination, Licensee shall promptly return or destroy all Confidential Information
and cease all use of the Software.

8. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of the State of
Delaware, without regard to its conflict of laws principles. Any disputes arising hereunder
shall be resolved exclusively in the state or federal courts of New Castle County, Delaware,
and each Party hereby submits to the personal jurisdiction of such courts.

9. NON-COMPETE
During the term of this Agreement and for a period of one (1) year thereafter, Licensee shall
not directly or indirectly engage in, or have any interest in, any person, firm, corporation,
or business that directly competes with Licensor's core software product offerings in the
enterprise data management sector within the continental United States.

10. DATA PROTECTION
Each Party shall comply with all applicable data protection and privacy laws, including but not
limited to GDPR, CCPA, and any other applicable regulations. Licensor shall implement appropriate
technical and organizational measures to protect personal data processed under this Agreement,
including encryption at rest and in transit, and shall not transfer personal data outside the
European Economic Area without appropriate safeguards.

11. WARRANTIES
Licensor warrants that the Software will perform substantially in accordance with the accompanying
documentation for a period of ninety (90) days following delivery. EXCEPT AS EXPRESSLY PROVIDED
HEREIN, THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING WITHOUT LIMITATION ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
OR NON-INFRINGEMENT.

12. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the Parties with respect to the subject
matter hereof and supersedes all prior negotiations, representations, warranties, and understandings.
This Agreement may not be amended except by a written instrument signed by authorized
representatives of both Parties.
"""

# ── Default categories ──────────────────────────────────────────────────────────
DEFAULT_CATEGORIES = [
    "Confidentiality",
    "Termination",
    "Governing Law",
    "Indemnification",
    "Limitation of Liability",
    "IP Ownership",
    "Payment Terms",
    "Non-Compete",
]

# ── Pre-rendered demo results ───────────────────────────────────────────────────
DEMO_RESULTS: dict[str, list[dict[str, Any]]] = {
    "Confidentiality": [
        {
            "text": "Each Party agrees to maintain in strict confidence all Confidential Information received from the other Party.",
            "status": "verified",
            "confidence": 0.95,
            "quote": "Each Party agrees to maintain in strict confidence all Confidential Information received from the other Party.",
            "reasoning": "Exact text found verbatim in Section 1 (Confidentiality). The clause is directly grounded in the source contract.",
        }
    ],
    "Termination": [
        {
            "text": "Either Party may terminate this Agreement upon thirty (30) days written notice to the other Party.",
            "status": "verified",
            "confidence": 0.92,
            "quote": "Either Party may terminate this Agreement upon thirty (30) days written notice to the other Party.",
            "reasoning": "Exact match found in Section 7 (Termination).",
        },
        {
            "text": "Licensor may terminate this Agreement immediately upon Licensee's material breach of any provision hereof.",
            "status": "verified",
            "confidence": 0.90,
            "quote": "Licensor may terminate this Agreement immediately upon Licensee's material breach of any provision hereof.",
            "reasoning": "Second termination clause found in Section 7, grounded verbatim in the source.",
        },
    ],
    "Governing Law": [
        {
            "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.",
            "status": "verified",
            "confidence": 0.97,
            "quote": "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.",
            "reasoning": "Governing law clause found verbatim in Section 8.",
        }
    ],
    "Indemnification": [
        {
            "text": "Licensor shall indemnify, defend, and hold harmless Licensee and its officers, directors, and employees from and against any third-party claims arising out of Licensor's infringement of any patent, copyright, or trade secret rights.",
            "status": "verified",
            "confidence": 0.91,
            "quote": "Licensor shall indemnify, defend, and hold harmless Licensee and its officers, directors, and employees from and against any third-party claims...",
            "reasoning": "Indemnification clause found in Section 6, grounded in the source contract.",
        }
    ],
    "Limitation of Liability": [
        {
            "text": "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES... EACH PARTY'S TOTAL CUMULATIVE LIABILITY SHALL NOT EXCEED THE FEES PAID BY LICENSEE IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.",
            "status": "verified",
            "confidence": 0.94,
            "quote": "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES...",
            "reasoning": "Standard limitation of liability clause found in Section 5.",
        }
    ],
    "IP Ownership": [
        {
            "text": "All intellectual property rights in and to the Software, including all modifications, enhancements, and derivative works, shall remain the exclusive property of Licensor.",
            "status": "verified",
            "confidence": 0.93,
            "quote": "All intellectual property rights in and to the Software, including all modifications, enhancements, and derivative works, shall remain the exclusive property of Licensor.",
            "reasoning": "IP ownership clause found verbatim in Section 4.",
        }
    ],
    "Payment Terms": [
        {
            "text": "Licensee agrees to pay Licensor an annual license fee of $50,000, due within 30 days of invoice.",
            "status": "verified",
            "confidence": 0.96,
            "quote": "Licensee agrees to pay Licensor an annual license fee of $50,000, due within 30 days of invoice.",
            "reasoning": "Payment terms clause found verbatim in Section 3.",
        },
        {
            "text": "Late payments shall accrue interest at 1.5% per month or the maximum rate permitted by law.",
            "status": "verified",
            "confidence": 0.88,
            "quote": "Late payments shall accrue interest at 1.5% per month or the maximum rate permitted by law, whichever is less.",
            "reasoning": "Late payment clause in Section 3 — near-exact match with minor trailing truncation.",
        },
    ],
    "Non-Compete": [
        {
            "text": "During the term of this Agreement and for a period of one (1) year thereafter, Licensee shall not directly or indirectly engage in any business that competes with Licensor's core software product offerings in the enterprise data management sector.",
            "status": "verified",
            "confidence": 0.88,
            "quote": "Licensee shall not directly or indirectly engage in, or have any interest in, any person, firm, corporation, or business that directly competes with Licensor's core software product offerings...",
            "reasoning": "Non-compete clause found in Section 9 with near-exact match.",
        }
    ],
}

DEMO_METRICS: dict[str, Any] = {
    "verified": 10,
    "rejected": 0,
    "total": 10,
    "cost": 0.0042,
    "avg_latency": 1230.0,
    "model": "gpt-4o-mini (demo)",
}


# ── Pipeline runner ─────────────────────────────────────────────────────────────

def _set_openai_key(api_key: str) -> None:
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        import litellm  # type: ignore[import]
        litellm.api_key = api_key
    except Exception:
        pass


def run_extraction_pipeline(
    contract_text: str,
    category_names: list[str],
    model: str,
    api_key: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run extraction → verification for each selected category."""
    _set_openai_key(api_key)

    from contractlens.models import ClauseCategory, Contract  # type: ignore[import]
    from contractlens.extraction.extractor import ClauseExtractor  # type: ignore[import]
    from contractlens.verification.verifier import SpanVerifier  # type: ignore[import]

    contract = Contract(
        contract_id=str(uuid.uuid4()),
        title="Uploaded Contract",
        text=contract_text,
    )

    cat_enums: list[ClauseCategory] = []
    for name in category_names:
        try:
            cat_enums.append(ClauseCategory(name))
        except ValueError:
            continue

    extractor = ClauseExtractor(default_model=model)
    verifier = SpanVerifier(default_model=model)

    all_results: dict[str, list[dict[str, Any]]] = {}
    total_cost = 0.0
    total_latency = 0.0
    call_count = 0
    verified_count = 0
    rejected_count = 0

    n = len(cat_enums)
    prog = st.progress(0.0, text="Starting…")
    status_box = st.empty()

    for i, cat in enumerate(cat_enums):
        prog.progress(i / n, text=f"Extracting {cat.value} ({i + 1}/{n})…")
        status_box.markdown(f"**Step 2 / 3 — Extraction** &nbsp;`{cat.value}`")

        try:
            spans = extractor.extract(contract.text, cat, model=model)
        except Exception as exc:
            status_box.error(f"Extraction error for {cat.value}: {exc}")
            all_results[cat.value] = []
            continue

        category_results: list[dict[str, Any]] = []
        for span in spans:
            prog.progress(
                (i + 0.6) / n,
                text=f"Verifying {cat.value} ({i + 1}/{n})…",
            )
            status_box.markdown(f"**Step 3 / 3 — Verification** &nbsp;`{cat.value}`")
            try:
                vr = verifier.verify(span, contract.text, model=model)
                total_cost += vr.cost_usd
                total_latency += vr.verification_time_ms
                call_count += 1
                is_ver = vr.status.value == "verified"
                verified_count += int(is_ver)
                rejected_count += int(not is_ver)
                category_results.append(
                    {
                        "text": span.text,
                        "status": vr.status.value,
                        "confidence": span.confidence,
                        "quote": vr.verification_quote or "",
                        "reasoning": vr.reasoning,
                    }
                )
            except Exception as exc:
                rejected_count += 1
                category_results.append(
                    {
                        "text": span.text,
                        "status": "rejected",
                        "confidence": span.confidence,
                        "quote": "",
                        "reasoning": f"Verification error: {exc}",
                    }
                )

        all_results[cat.value] = category_results

    prog.progress(1.0, text="Complete!")
    status_box.success("Pipeline finished.")

    metrics: dict[str, Any] = {
        "verified": verified_count,
        "rejected": rejected_count,
        "total": verified_count + rejected_count,
        "cost": total_cost,
        "avg_latency": total_latency / max(call_count, 1),
        "model": model,
    }
    return all_results, metrics


# ── Rendering helpers ───────────────────────────────────────────────────────────

def _pipeline_steps(active: int) -> None:
    steps = ["📄 Index", "🔍 Extract", "✅ Verify", "✨ Done"]
    parts: list[str] = []
    for i, label in enumerate(steps, 1):
        if i < active:
            parts.append(f'<span class="step-done">{label}</span>')
        elif i == active:
            parts.append(f'<span class="step-active">{label} ◀</span>')
        else:
            parts.append(f'<span class="step-todo">{label}</span>')
    st.markdown(" &nbsp;→&nbsp; ".join(parts), unsafe_allow_html=True)
    st.write("")


def render_results(
    results: dict[str, list[dict[str, Any]]],
    metrics: dict[str, Any],
    is_demo: bool = False,
) -> None:
    if is_demo:
        st.info(
            "📊 **Sample results** on the built-in contract. "
            "Enter your OpenAI API key in the sidebar to run live extraction on any document.",
            icon="ℹ️",
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("✅ Verified", metrics["verified"])
    c2.metric("❌ Rejected", metrics["rejected"])
    c3.metric("📄 Total spans", metrics["total"])
    c4.metric("💰 Cost", f"${metrics['cost']:.4f}")
    c5.metric("⚡ Avg latency", f"{metrics['avg_latency']:.0f} ms")

    st.divider()

    any_result = False
    for cat_name, spans in results.items():
        if not spans:
            continue
        any_result = True
        n_ver = sum(1 for s in spans if s["status"] == "verified")
        n_rej = len(spans) - n_ver
        with st.expander(f"**{cat_name}** — {n_ver} verified, {n_rej} rejected", expanded=True):
            for span in spans:
                is_ver = span["status"] == "verified"
                badge = (
                    '<span class="badge-verified">✓ VERIFIED</span>'
                    if is_ver
                    else '<span class="badge-rejected">✗ REJECTED</span>'
                )
                card_cls = "clause-card" if is_ver else "clause-card rejected"
                conf_pct = int(span["confidence"] * 100)
                st.markdown(
                    f'{badge} &nbsp; <small style="color:#6b7280">confidence {conf_pct}%</small>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="{card_cls}">{span["text"]}</div>',
                    unsafe_allow_html=True,
                )
                if span.get("quote"):
                    st.caption(f"🔗 Source quote: _{span['quote'][:280]}_")
                with st.expander("Verifier reasoning", expanded=False):
                    st.caption(span["reasoning"])
                st.write("")

    if not any_result:
        st.warning("No clauses were extracted. Try adding more categories or a longer contract.")


def render_about() -> None:
    st.markdown(
        """
## About ContractLens

ContractLens is a production-style system for extracting legal clause spans from contracts
across **41 CUAD clause categories**, built on the
[CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad)
(510 contracts, 13K+ expert annotations).

### Pipeline

```
Contract Text
      │
      ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Retrieval  │──▶│  Extraction │──▶│ Verification│──▶│   Result    │
│  (ChromaDB) │   │  (LiteLLM)  │   │ (LLM-judge) │   │             │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                        │                  │
                        └──── retry ◀──────┘   (if >50% rejected, max 2×)
```

Orchestrated via **LangGraph** with a retry loop: if the LLM-as-judge rejects
more than half of the extracted spans, the extraction node reruns with expanded context.

### Model Benchmark (CUAD test set)

| Model | Span F1 | Cost / contract | Avg latency |
|-------|---------|-----------------|-------------|
| GPT-4o | 0.84 | $0.045 | 1,250 ms |
| GPT-4o-mini | 0.80 | $0.012 | 380 ms |
| Llama 3 8B (LoRA fine-tuned) | 0.73 | $0.001 | 2,100 ms |

The LoRA-fine-tuned Llama 3 8B matches frontier quality on common clause types
at a fraction of the inference cost.

### Error Taxonomy

| Error Type | Share |
|------------|-------|
| `OFFSET_ERROR` | 23 % |
| `VERIFICATION_FAILED` | 18 % |
| `PARTIAL_EXTRACTION` | 15 % |
| `RETRIEVAL_NOISE` | 12 % |
| `WRONG_CATEGORY` | 9 % |

### Tech Stack

**Python** · **LangGraph** · **ChromaDB** · **Azure AI Foundry** · **FastAPI** ·
**Docker** · **LiteLLM** · **Streamlit**

---

[GitHub →](https://github.com/charanyellanki/ContractLens) &nbsp;|&nbsp; Built by Charan Yellanki
"""
    )


# ── Sidebar ─────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, list[str], bool, bool]:
    with st.sidebar:
        st.markdown("## ⚖️ ContractLens")
        st.caption("Legal clause extraction · LangGraph · LLM-as-judge")
        st.divider()

        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            st.success("API key loaded from environment.", icon="🔑")
            api_key = env_key
        else:
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-…",
                help="Used only for this session — never stored.",
            )

        st.divider()

        model: str = st.radio(  # type: ignore[assignment]
            "Model",
            options=["gpt-4o-mini", "gpt-4o"],
            captions=["Fast & cheap · $0.15 / 1M tokens", "Higher accuracy · $5 / 1M tokens"],
        )

        st.divider()

        from contractlens.models import ClauseCategory  # type: ignore[import]
        all_cats = sorted(c.value for c in ClauseCategory)
        selected: list[str] = st.multiselect(
            "Clause types to extract",
            options=all_cats,
            default=DEFAULT_CATEGORIES,
        )

        st.divider()

        extract_clicked = st.button(
            "🔍 Extract Clauses", type="primary", use_container_width=True
        )
        demo_clicked = st.button("📊 Load Sample Results", use_container_width=True)

        if "metrics" in st.session_state:
            st.divider()
            m: dict[str, Any] = st.session_state["metrics"]
            st.markdown(f"**Last run — `{m['model']}`**")
            ca, cb = st.columns(2)
            ca.metric("✅", m["verified"])
            cb.metric("❌", m["rejected"])
            st.metric("💰 Cost", f"${m['cost']:.4f}")
            st.metric("⚡ Latency", f"{m['avg_latency']:.0f} ms")

    return api_key, model, selected, extract_clicked, demo_clicked


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key, model, selected_categories, extract_clicked, demo_clicked = render_sidebar()

    st.title("⚖️ ContractLens")
    st.markdown(
        "**Legal contract clause extraction** · LangGraph orchestration · LLM-as-judge verification  \n"
        "Built on the **CUAD dataset** — 510 contracts, 13K+ expert annotations, 41 clause types."
    )

    tab_extract, tab_about = st.tabs(["📄 Extract", "ℹ️ About"])

    # ── Extract tab ─────────────────────────────────────────────────────────────
    with tab_extract:
        col_text, col_upload = st.columns([3, 1])

        with col_text:
            # Populate text area from session state (sample load or file upload)
            prefill = st.session_state.pop("prefill_text", st.session_state.get("contract_text", ""))
            contract_text: str = st.text_area(
                "Paste contract text",
                value=prefill,
                height=280,
                placeholder="Paste the full text of a legal contract here…",
            )
            st.session_state["contract_text"] = contract_text

        with col_upload:
            st.markdown("**Or upload a file**")
            uploaded = st.file_uploader("Upload .txt", type=["txt"], label_visibility="collapsed")
            if uploaded is not None:
                try:
                    loaded = uploaded.read().decode("utf-8")
                    st.session_state["prefill_text"] = loaded
                    st.success(f"Loaded `{uploaded.name}`")
                    st.rerun()
                except Exception:
                    st.error("Could not read file.")

            st.write("")
            if st.button("Load sample contract", use_container_width=True):
                st.session_state["prefill_text"] = SAMPLE_CONTRACT
                st.rerun()

        st.divider()

        # ── Button actions ─────────────────────────────────────────────────────
        if demo_clicked:
            st.session_state["results"] = DEMO_RESULTS
            st.session_state["metrics"] = DEMO_METRICS
            st.session_state["is_demo"] = True

        if extract_clicked:
            effective_text = st.session_state.get("contract_text", "").strip()
            if not api_key:
                st.error("Enter your OpenAI API key in the sidebar.")
            elif not effective_text:
                st.error("Paste a contract or upload a file first.")
            elif not selected_categories:
                st.error("Select at least one clause type in the sidebar.")
            else:
                st.session_state["is_demo"] = False
                _pipeline_steps(2)
                try:
                    results, metrics = run_extraction_pipeline(
                        effective_text, selected_categories, model, api_key
                    )
                    st.session_state["results"] = results
                    st.session_state["metrics"] = metrics
                    _pipeline_steps(4)
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc(), language="text")

        # ── Results ────────────────────────────────────────────────────────────
        if "results" in st.session_state:
            render_results(
                st.session_state["results"],
                st.session_state["metrics"],
                is_demo=st.session_state.get("is_demo", False),
            )

    # ── About tab ───────────────────────────────────────────────────────────────
    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
