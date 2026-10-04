"""Optional demo UI. Start the API first (`make serve`), then `make ui`."""
import os

import httpx
import streamlit as st

API = os.environ.get("RECSYS_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid Recommender", layout="wide")
st.title("Hybrid Recommendation Engine")


@st.cache_data(ttl=30)
def get(path: str, **params):
    r = httpx.get(f"{API}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


try:
    info = get("/models")
except Exception as exc:
    st.error(f"API not reachable at {API}: {exc}")
    st.stop()

if not info["ready"]:
    st.warning("No model loaded. Run `python -m training.train` and restart the API.")
    st.stop()

tab_recs, tab_similar, tab_models = st.tabs(["Recommendations", "Similar items", "Models & metrics"])

with tab_recs:
    c1, c2, c3 = st.columns(3)
    user_id = c1.number_input("User id", min_value=1, value=1, step=1)
    model = c2.selectbox("Model", [m["name"] for m in info["models"]], index=len(info["models"]) - 1)
    k = c3.slider("K", 1, 30, 10)
    try:
        recs = get(f"/recommendations/{int(user_id)}", k=k, model=model)
        st.caption(f"source: **{recs['source']}** · model version {recs['model_version']}")
        st.dataframe(
            [{"#": i["rank"], "title": i["title"], "genres": ", ".join(i["genres"]), "score": i["score"],
              "why": i["explanation"]} for i in recs["items"]],
            use_container_width=True, hide_index=True,
        )
        with st.expander("User history"):
            hist = get(f"/users/{int(user_id)}/history", limit=30)
            st.write(f"{hist['total_interactions']} interactions · top genres: {', '.join(hist['top_genres']) or '-'}")
            st.dataframe(hist["interactions"], use_container_width=True, hide_index=True)
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.json().get("detail", str(exc)))

with tab_similar:
    item_id = st.number_input("Item id", min_value=1, value=1, step=1)
    method = st.radio("Similarity", ["hybrid", "content", "collaborative"], horizontal=True)
    try:
        sim = get(f"/items/{int(item_id)}/similar", k=10, method=method)
        st.dataframe(sim["items"], use_container_width=True, hide_index=True)
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.json().get("detail", str(exc)))

with tab_models:
    st.json(info["hybrid_weights"])
    if st.button("Run /evaluate (re-computes the comparison)"):
        rep = httpx.post(f"{API}/evaluate", json={"k_values": [10]}, timeout=300).json()
        st.dataframe(
            [{"model": n, **{k: v for k, v in r["metrics"].items() if k.endswith("@10")}} for n, r in rep["results"].items()],
            use_container_width=True, hide_index=True,
        )
