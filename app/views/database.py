"""Page 4: the similarity cache inside SQLite (R*Tree on a PCA-3 projection, exact k-NN)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import fmt_angles, setup

from qws.db import similarity as sim
from qws.features import FEATURE_NAMES
from qws.features import vector as feature_vector


def render():
    w = setup()
    st.title("کش در پایگاه داده")
    st.markdown(
        "هر گراف حل‌شده با ۱۶ ویژگی‌اش (اندازه، وزن‌ها، طیف لاپلاسی، ...) در SQLite ذخیره می‌شود. "
        "برای پیدا کردن گراف‌های شبیه، «سایهٔ» ۳بعدی هر نقطه (PCA) در یک R*Tree نگه داشته می‌شود. "
        "سایهٔ دو نقطه هیچ‌وقت از خودشان دورتر نیست، پس جست‌وجو **دقیق** است و هیچ همسایه‌ای جا نمی‌افتد.")
    con, ids = sim_db(w)
    k = st.slider("چند همسایه؟ (k)", 1, 8, 3)
    x = feature_vector(w.graph)
    res = sim.knn(con, x, k, p=w.p, exclude={ids[-1]})
    P = sim.load_projection(con)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("گراف‌های داخل پایگاه داده", len(ids))
    c2.metric("واریانس حفظ‌شده با PCA-3", f"{P.explained:.0%}")
    c3.metric("نامزدهای بررسی‌شده", res.candidates)
    c4.metric("دورهای جست‌وجو", res.rounds)

    rows = []
    for gid, d in zip(res.ids, res.dist):
        th = sim.cached_angles(con, gid, w.p)
        n = con.execute("SELECT n_nodes, split FROM graph WHERE graph_id=?", (gid,)).fetchone()
        rows.append({"graph_id": gid, "فاصله": round(d, 3), "n": n[0],
                     "منبع": "آموزشی" if n[1] == "train" else "همین جریان",
                     "زاویه‌ها": fmt_angles((th[:w.p], th[w.p:]))})
    st.subheader("نزدیک‌ترین گراف‌ها به گراف فعلی")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    Y = np.array([P.project(feature_vector(g)) for g, _, _ in w.solved])
    yq = P.project(x)
    near = np.isin(ids, res.ids)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Y[~near, 0], y=Y[~near, 1], mode="markers", name="گراف‌های ذخیره‌شده",
                             marker=dict(size=9, color="#9aa0a6")))
    fig.add_trace(go.Scatter(x=Y[near, 0], y=Y[near, 1], mode="markers", name="همسایه‌های پیدا‌شده",
                             marker=dict(size=12, color="#D55E00")))
    fig.add_trace(go.Scatter(x=[yq[0]], y=[yq[1]], mode="markers", name="گراف فعلی",
                             marker=dict(size=16, symbol="star", color="#0072B2")))
    r = res.radius
    fig.add_shape(type="rect", x0=yq[0] - r, x1=yq[0] + r, y0=yq[1] - r, y1=yq[1] + r,
                  line=dict(dash="dot", color="#0072B2"))
    fig.update_layout(height=460, xaxis_title="مؤلفهٔ اصلی ۱", yaxis_title="مؤلفهٔ اصلی ۲",
                      yaxis=dict(scaleanchor="x"), margin=dict(l=10, r=10, t=30, b=110), legend=dict(orientation="h", y=-0.38, x=0))
    st.subheader("سایهٔ گراف‌ها روی دو جهت اصلی")
    st.plotly_chart(fig, width="stretch")
    st.caption("مربع نقطه‌چین = جعبهٔ جست‌وجوی R*Tree در آخرین دور (برش دوبعدی از مکعب سه‌بعدی).")

    with st.expander("پرس‌وجوی SQL و برنامهٔ اجرای آن (EXPLAIN QUERY PLAN)"):
        q = sim.cube_sql(w.p)
        st.code(q, language="sql")
        box = [v for c in yq for v in (c - r, c + r)]
        plan = [row[3] for row in con.execute("EXPLAIN QUERY PLAN " + q, [w.p] + box)]
        st.code("\n".join(plan), language="text")
        st.caption("«VIRTUAL TABLE INDEX» یعنی R*Tree شش شرط بازه را جواب داده است، نه جست‌وجوی کامل جدول.")
    with st.expander("۱۶ ویژگی گراف فعلی"):
        st.dataframe(pd.DataFrame({"ویژگی": FEATURE_NAMES, "مقدار": np.round(x, 4)}), hide_index=True)


def sim_db(w):
    """Rebuild the in-memory DB only when the cache changed."""
    from qws.demo import build_db
    key = (id(w), len(w.solved))
    if st.session_state.get("db_key") != key:
        st.session_state.db = build_db(w)
        st.session_state.db_key = key
    return st.session_state.db
