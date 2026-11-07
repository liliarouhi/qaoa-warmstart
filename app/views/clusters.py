"""Page 1: the orders on the map and the two groups chosen by Max-Cut."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import SIDE_COLORS, order_buttons, setup

from qws.demo import best_cut, cut_value, within_cluster_distance


def render():
    w = setup()
    st.title("سفارش‌ها را دو گروه کن")
    st.markdown(
        "هر نقطه یک سفارش است. وزن هر یال فاصلهٔ دو سفارش است. **بیشینه کردن برش** یعنی دو گروهی بسازیم که "
        "سفارش‌های داخل هر گروه به هم نزدیک باشند، چون جمع کل فاصله‌ها ثابت است و برش = کل − فاصلهٔ داخل گروه‌ها. "
        "این همان مرحلهٔ «اول خوشه‌بندی، بعد مسیریابی» است.")
    order_buttons(w, "clusters")

    g = w.graph
    best, sides, total = best_cut(g)
    n = g.n
    labels = [str(i + 1) for i in range(n)]
    mine = st.multiselect("حدس خودت: کدام سفارش‌ها در گروه نارنجی باشند؟", labels, default=[])
    my_sides = np.array([1 if str(i + 1) in mine else 0 for i in range(n)])
    show_best = st.toggle("بهترین برش را نشانم بده (Max-Cut دقیق، جست‌وجوی کامل)", value=not mine)
    shown = sides if show_best else my_sides

    fig = go.Figure()
    iu, ju = g.edges[:, 0], g.edges[:, 1]
    for u, v in zip(iu, ju):  # draw only the cut edges, to keep the picture readable
        if shown[u] != shown[v]:
            fig.add_trace(go.Scatter(x=w.points[[u, v], 0], y=w.points[[u, v], 1], mode="lines",
                                     line=dict(color="rgba(150,150,150,0.25)", width=1), hoverinfo="skip",
                                     showlegend=False))
    for s in (0, 1):
        m = shown == s
        fig.add_trace(go.Scatter(x=w.points[m, 0], y=w.points[m, 1], mode="markers+text",
                                 text=[labels[i] for i in np.where(m)[0]], textposition="top center",
                                 marker=dict(size=16, color=SIDE_COLORS[s], line=dict(width=1, color="white")),
                                 name=["گروه آبی", "گروه نارنجی"][s]))
    fig.update_layout(height=520, xaxis=dict(range=[-0.05, 1.05], title="x"),
                      yaxis=dict(range=[-0.05, 1.05], title="y", scaleanchor="x"),
                      margin=dict(l=10, r=10, t=30, b=110), legend=dict(orientation="h", y=-0.38, x=0))
    c1, c2 = st.columns([3, 1])
    c1.plotly_chart(fig, width="stretch")
    with c2:
        val = cut_value(g, shown)
        st.metric("مقدار برش", f"{val:.3f}")
        st.metric("بهترین برش ممکن", f"{best:.3f}")
        st.metric("نسبت به بهترین", f"{val / best:.1%}")
        st.metric("فاصلهٔ داخل گروه‌ها", f"{within_cluster_distance(g, shown):.3f}")
        st.caption(f"جمع همهٔ فاصله‌ها: {total:.3f} = برش + فاصلهٔ داخل گروه‌ها")
    st.info(f"برای پیدا کردن بهترین برش، کامپیوتر همهٔ `2^{n} = {2**n:,}` حالت را امتحان کرد. "
            "با هر سفارش اضافه این عدد دو برابر می‌شود. QAOA قرار است این جست‌وجو را روی کامپیوتر کوانتومی انجام دهد.")
