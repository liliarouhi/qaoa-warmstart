"""Page 5: how far do the optimal angles move when one order is added? (Conjecture E, E14)"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from common import setup

from qws.demo import E14, shift_experiment


def render():
    setup()
    st.title("نظریه: زاویه‌ها چقدر جابه‌جا می‌شوند؟")
    st.markdown(
        "شروع گرم وقتی خوب کار می‌کند که بهترین زاویه‌ها با یک سفارش جدید **کم** جابه‌جا شوند. "
        "آزمایش E14 نشان داد جابه‌جایی نسبی γ تقریباً مثل 1/n کم می‌شود (شیب 0.99−). "
        "اثبات کامل باز است (Conjecture E). اینجا می‌توانی خودت امتحان کنی؛ "
        "برای سرعت از فرمول بستهٔ p=1 استفاده می‌شود، پس n می‌تواند بزرگ باشد.")
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("تعداد سفارش n", 8, 60, 20, step=4)
        reps = st.slider("چند نمونهٔ تصادفی", 1, 10, 3)
        go_btn = st.button("آزمایش کن", type="primary")
    if go_btn:
        rng = np.random.default_rng()
        out = [shift_experiment(n, rng) for _ in range(reps)]
        st.session_state.theory = (n, out)
    fig = go.Figure()
    xs = np.array(E14["n"], float)
    fig.add_trace(go.Scatter(x=xs, y=E14["rel_gamma_shift"], mode="lines+markers", name="میانهٔ E14 (۴۰ گراف برای هر n)",
                             line=dict(color="#0072B2", width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=E14["rel_gamma_shift"][0] * xs[0] / xs, mode="lines", name="خط مرجع 1/n",
                             line=dict(color="#999999", dash="dot")))
    if "theory" in st.session_state:
        tn, out = st.session_state.theory
        ys = [o["rel_gamma_shift"] for o in out]
        fig.add_trace(go.Scatter(x=[tn] * len(ys), y=ys, mode="markers", name="آزمایش‌های تو",
                                 marker=dict(size=11, color="#D55E00")))
        with c1:
            st.metric("میانهٔ جابه‌جایی نسبی γ در آزمایش تو", f"{np.median(ys):.4f}")
            o = out[0]
            st.caption(f"نمونهٔ اول: γ* از {o['before'][0]:.4f} به {o['after'][0]:.4f}، "
                       f"β* از {o['before'][1]:.4f} به {o['after'][1]:.4f}")
    fig.update_layout(height=460, xaxis_type="log", yaxis_type="log", xaxis_title="n (تعداد سفارش)",
                      yaxis_title="جابه‌جایی نسبی γ*", margin=dict(l=10, r=10, t=30, b=110), legend=dict(orientation="h", y=-0.38, x=0))
    c2.plotly_chart(fig, width="stretch")
    st.markdown(
        "**یافتهٔ جانبی:** خود γ* مثل 1/n کم نمی‌شود، ولی حاصل‌ضرب n·γ*·β* تقریباً ثابت است (حدود 1.8). "
        "یعنی بهترین نقطه کف یک «درهٔ کم‌عمق» است. جابه‌جا شدن خود دره O(1/n) است و اثبات شده (Lemma E3)؛ "
        "حرکت در طول دره هنوز حدس است.")
