"""Page 2: the two QAOA knobs (gamma, beta) and the landscape F(gamma, beta), p = 1."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import COLORS, fmt_angles, setup

from qws.demo import best_cut, landscape_p1
from qws.simulator import QAOASimulator


@st.cache_data(show_spinner="محاسبهٔ نقشه با فرمول بسته (Lemma E1)...")
def _land(points_bytes, n):
    from qws.demo import graph_of
    pts = np.frombuffer(points_bytes).reshape(n, 2)
    return landscape_p1(graph_of(pts))


def render():
    w = setup()
    st.title("دو پیچ QAOA را بچرخان")
    st.markdown(
        "QAOA با p=1 فقط دو عدد دارد: γ و β. هر جفت (γ, β) یک مقدار برش مورد انتظار F می‌دهد. "
        "نقشهٔ رنگی زیر F را برای همهٔ جفت‌ها نشان می‌دهد. کار بهینه‌ساز این است که روشن‌ترین نقطه را پیدا کند. "
        "**مسئلهٔ پروژه:** از کجا شروع کنیم که زود به آنجا برسیم؟")
    g = w.graph
    gs, bs, F = _land(w.points.tobytes(), len(w.points))
    best, _, _ = best_cut(g)
    R = F / best
    lab = w.prev_theta
    p_label = w.p
    c1, c2 = st.columns([3, 1])
    with c2:
        gamma = st.slider("γ (پیچ اول)", float(gs[0]), float(gs[-1]), float(gs[len(gs) // 3]), step=float(gs[1] - gs[0]))
        beta = st.slider("β (پیچ دوم)", -0.785, 0.785, -0.3, step=0.01)
        sim = QAOASimulator(g)
        f = sim.expectation([gamma], [beta])
        st.metric("F(γ, β) / بهترین برش", f"{f / best:.3f}")
        probs = sim.probabilities([gamma], [beta])
        top = np.isclose(sim.c, sim.c_max)
        st.metric("احتمال دیدن بهترین برش در یک اندازه‌گیری", f"{probs[top].sum():.1%}")
        st.caption(f"شروع تصادفی (همه حالت‌ها هم‌احتمال): {top.sum() / 2**g.n:.2%}")
    fig = go.Figure(go.Heatmap(x=bs, y=gs, z=R, colorscale="Viridis", colorbar=dict(title="F / Cmax"),
                               hovertemplate="β=%{x:.3f}<br>γ=%{y:.3f}<br>F/Cmax=%{z:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[beta], y=[gamma], mode="markers", name="پیچ‌های تو",
                             marker=dict(size=14, color="white", line=dict(width=2, color="black"))))
    if p_label == 1:
        fig.add_trace(go.Scatter(x=[float(lab[1][0])], y=[float(lab[0][0])], mode="markers", name="بهترین (برچسب)",
                                 marker=dict(size=14, symbol="star", color="red")))
        fig.add_trace(go.Scatter(x=[float(w.fixed_theta[1][0])], y=[float(w.fixed_theta[0][0])], mode="markers",
                                 name="زاویهٔ ثابت", marker=dict(size=12, symbol="diamond", color=COLORS["fixed"])))
    fig.update_layout(height=540, xaxis_title="β", yaxis_title="γ", margin=dict(l=10, r=10, t=30, b=110),
                      legend=dict(orientation="h", y=-0.38, x=0))
    c1.plotly_chart(fig, width="stretch")
    if p_label == 1:
        st.caption(f"بهترین زاویه‌ها برای این گراف: {fmt_angles(lab)} · زاویهٔ ثابت (میانهٔ گراف‌های آموزشی): "
                   f"{fmt_angles(w.fixed_theta)}")
    else:
        st.caption("نقشه برای p=1 است (دو بعد). برای p=2 فضای زاویه‌ها ۴ بعدی است و فقط در صفحهٔ ۳ دیده می‌شود.")
    st.markdown("**تقارن‌ها:** β دوره‌ای با دورهٔ `π/2` است، پس فقط بازهٔ `[-π/4, π/4)` نشان داده شده. "
                "با جابه‌جایی `(γ, β) → (-γ, -β)` مقدار F عوض نمی‌شود، پس `γ ≥ 0` کافی است (تقارن وارونگی زمان).")
