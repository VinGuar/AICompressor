"""
ContextLite — Streamlit UI.
Run: python -m streamlit run app.py
"""

import re
import streamlit as st

st.set_page_config(page_title="ContextLite", page_icon="⚡", layout="wide")


# ── Cache model load — happens once, survives reruns ─────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (one-time, ~5s)...")
def load_model():
    from contextlite.embedder import get_model
    return get_model()


# ── Cache optimize results — same inputs = instant replay ────────────────────
@st.cache_data(show_spinner=False)
def run_optimize(chunks_tuple, query, token_budget, relevance_threshold, dedup_threshold, mmr_lambda):
    from contextlite.pipeline import optimize
    return optimize(
        list(chunks_tuple), query,
        token_budget=token_budget,
        relevance_threshold=relevance_threshold,
        dedup_threshold=dedup_threshold,
        mmr_lambda=mmr_lambda,
    )


# Pre-warm model on startup so first Optimize click is fast
load_model()

# ── Demo data ─────────────────────────────────────────────────────────────────
DEMO_CHUNKS = [
    "Our pricing starts at $499/month for the Starter plan. The Professional plan is $999/month. Enterprise is $2,499/month. Annual billing gives a 20% discount on all plans. A 14-day free trial is available with no credit card required.",
    "The Starter plan includes 1,000 API requests per minute. Professional allows 10,000 requests per minute. Enterprise has no rate limits. All plans include REST and GraphQL API access.",
    "Our company was founded in 2018 with offices in Austin, New York, and London. The CEO founded two previous companies. We have raised $24M in Series B funding led by Sequoia Capital. Our engineering team is distributed across 12 time zones.",
    "Pricing for the platform begins at $499 per month on the Starter tier. Professional tier costs $999 monthly. The Enterprise tier is priced at $2,499/month. Customers on annual plans receive a 20% discount.",
    "Customer support is available 24/7 via email and live chat. Support tickets are resolved within 4 hours on average. We have a 98% customer satisfaction score. The mobile app is available on iOS and Android.",
]
DEMO_QUERY = "What are the pricing plans and API rate limits?"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ ContextLite")
st.caption(
    "Paste the chunks your RAG system retrieved. ContextLite filters, deduplicates, "
    "and reranks them — then packs the best sentences into your token budget. No LLM calls."
)
st.divider()

# ── Sidebar: settings ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    st.markdown("#### Token Budget")
    token_budget = st.slider(
        "Max output tokens", min_value=64, max_value=4096, value=512, step=64,
        label_visibility="collapsed",
    )
    st.caption(
        f"Output context capped at **{token_budget} tokens**. "
        "Lower = more aggressive compression. Raise this if the output is cutting off useful content."
    )

    st.markdown("---")
    st.markdown("#### Relevance Filter")
    relevance_threshold = st.slider(
        "Relevance threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        label_visibility="collapsed",
    )
    if relevance_threshold < 0.2:
        st.caption("**Very loose** — keeps most sentences, even weakly related ones.")
    elif relevance_threshold < 0.35:
        st.caption("**Balanced** — keeps sentences with clear connection to the query.")
    else:
        st.caption("**Strict** — only keeps sentences that closely match the query. May miss context.")

    st.markdown("---")
    st.markdown("#### Duplicate Removal")
    dedup_threshold = st.slider(
        "Dedup threshold", min_value=0.5, max_value=1.0, value=0.85, step=0.05,
        label_visibility="collapsed",
    )
    if dedup_threshold > 0.92:
        st.caption("**Only exact duplicates** — two sentences must be nearly identical to be merged.")
    elif dedup_threshold > 0.78:
        st.caption("**Near-duplicates** — removes sentences that say the same thing in different words.")
    else:
        st.caption("**Aggressive** — removes loosely similar sentences. May remove useful variation.")

    st.markdown("---")
    st.markdown("#### Diversity (MMR)")
    mmr_lambda = st.slider(
        "MMR lambda", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        label_visibility="collapsed",
    )
    if mmr_lambda > 0.85:
        st.caption("**Pure relevance** — picks the most query-relevant sentences, even if repetitive.")
    elif mmr_lambda > 0.55:
        st.caption("**Balanced** — favours relevance but avoids picking similar sentences back-to-back.")
    else:
        st.caption("**Diverse** — prioritises covering different subtopics, even if less directly relevant.")

    st.markdown("---")
    st.markdown(
        "<small>Settings persist across runs. Results are cached — identical inputs return instantly.</small>",
        unsafe_allow_html=True,
    )

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input")

    if st.button("Load demo data", type="secondary", use_container_width=True):
        st.session_state["chunks_text"] = "\n\n---\n\n".join(DEMO_CHUNKS)
        st.session_state["query_text"] = DEMO_QUERY

    query = st.text_input(
        "Query",
        value=st.session_state.get("query_text", ""),
        placeholder="What question should the LLM answer from these chunks?",
    )

    chunks_raw = st.text_area(
        "Chunks  _(separate with a blank line or `---`)_",
        value=st.session_state.get("chunks_text", ""),
        height=360,
        placeholder="Paste your RAG-retrieved chunks here...",
    )

    n_chunks = len([c for c in re.split(r"\n---\n|\n{2,}", chunks_raw.strip()) if c.strip()]) if chunks_raw.strip() else 0
    if n_chunks:
        st.caption(f"{n_chunks} chunk{'s' if n_chunks != 1 else ''} detected")

    optimize_clicked = st.button("Optimize", type="primary", use_container_width=True)

with right:
    st.subheader("Output")

    if optimize_clicked:
        if not query.strip():
            st.error("Enter a query first.")
        elif not chunks_raw.strip():
            st.error("Paste at least one chunk.")
        else:
            raw_chunks = re.split(r"\n---\n|\n{2,}", chunks_raw.strip())
            chunks = [c.strip() for c in raw_chunks if c.strip()]

            with st.spinner("Optimizing..."):
                try:
                    result = run_optimize(
                        tuple(chunks), query,
                        token_budget, relevance_threshold, dedup_threshold, mmr_lambda,
                    )
                    st.session_state["result"] = result
                    st.session_state["original_chunks"] = chunks
                except Exception as e:
                    st.error(f"Error: {e}")

    result = st.session_state.get("result")

    if result:
        before = result["token_estimate_before"]
        after = result["token_estimate_after"]
        saved = before - after
        pct = round(result["compression_ratio"] * 100, 1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Before", f"{before:,} tok")
        m2.metric("After", f"{after:,} tok", delta=f"-{saved:,}", delta_color="normal")
        m3.metric("Reduction", f"{pct}%")
        m4.metric("Kept", f"{len(result['kept_sentences'])} sent")

        st.divider()

        tab_opt, tab_compare, tab_detail = st.tabs(["Optimized context", "Before / After", "What was removed"])

        with tab_opt:
            st.text_area(
                "optimized",
                value=result["optimized_context"],
                height=220,
                label_visibility="collapsed",
            )
            for line in result["explanation"]:
                st.caption(f"• {line}")

        with tab_compare:
            c1, c2 = st.columns(2)
            original_text = "\n\n".join(st.session_state.get("original_chunks", []))
            with c1:
                st.markdown(f"**Original** · {before:,} tokens")
                st.text_area("original", value=original_text, height=260, label_visibility="collapsed")
            with c2:
                st.markdown(f"**Optimized** · {after:,} tokens")
                st.text_area("optimized2", value=result["optimized_context"], height=260, label_visibility="collapsed")

        with tab_detail:
            kept_col, removed_col = st.columns(2)
            with kept_col:
                st.markdown(f"**Kept ({len(result['kept_sentences'])})**")
                for s in result["kept_sentences"]:
                    st.markdown(
                        f"<div style='background:#e8f5e9;border-radius:4px;padding:6px 10px;margin:4px 0;font-size:0.85em'>{s}</div>",
                        unsafe_allow_html=True,
                    )
            with removed_col:
                st.markdown(f"**Removed ({len(result['removed_sentences'])})**")
                for s in result["removed_sentences"]:
                    st.markdown(
                        f"<div style='background:#fce4ec;border-radius:4px;padding:6px 10px;margin:4px 0;font-size:0.85em'>{s}</div>",
                        unsafe_allow_html=True,
                    )
    else:
        st.info("Results will appear here. Load the demo data to try it immediately.")
