"""Page 3: a new order arrives; five ways to choose the starting angles race to the optimum."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import COLORS, METHOD_LABELS_FA, METHODS, order_buttons, setup

EXPLAIN = {
    "random": "هر بار از یک نقطهٔ تصادفی شروع می‌کند.",
    "fixed": "همیشه همان زاویه‌ها: میانهٔ جواب گراف‌های آموزشی.",
    "previous": "جواب گراف قبلی (قبل از آمدن سفارش) را دوباره استفاده می‌کند.",
    "newton": "از جواب قبلی یک گام Newton روی گراف جدید برمی‌دارد (Proposition D). دقیق است ولی Hessian هزینه دارد.",
    "cache": "شبیه‌ترین گراف‌های حل‌شده را از کش پیدا می‌کند و بهترینِ ۳ تا را امتحان می‌کند (۳ ارزیابی).",
}


def render():
    w = setup()
    st.title("سفارش جدید و شروع گرم")
    st.markdown(
        "هر بار که سفارشی می‌آید یا تحویل می‌شود، گراف کمی عوض می‌شود و QAOA باید دوباره بهینه شود. "
        "پنج روش برای انتخاب نقطهٔ شروع با هم مسابقه می‌دهند. هزینه = تعداد اجرای مدار تا رسیدن به ۹۹٫۹٪ بهترین مقدار "
        "(circuit-equivalents، همان معیار ETT پایان‌نامه).")
    order_buttons(w, "stream")
    if not w.history:
        st.info("هنوز رویدادی نیست. دکمهٔ «سفارش جدید» را بزن.")
        return

    h = w.history[-1]
    st.subheader(f"آخرین رویداد: {'سفارش جدید' if h['event'] == 'add' else 'تحویل سفارش'} · {h['n']} سفارش · p={w.p}")
    rows = []
    for m in METHODS:
        r = h["methods"][m]
        rows.append({"روش": METHOD_LABELS_FA[m], "هزینهٔ کل": r["total"], "کیفیت نقطهٔ شروع": round(r["start_quality"], 4),
                     "هزینهٔ شروع": r["start_cost"], "هزینهٔ بهینه‌ساز": r["ett"],
                     "به هدف رسید؟": "بله" if r["reached"] else "خیر (بودجه تمام شد)"})
    df = pd.DataFrame(rows)
    c1, c2 = st.columns([1, 1])
    c1.dataframe(df, hide_index=True, width="stretch")
    with c1:
        for m in METHODS:
            st.caption(f"**{METHOD_LABELS_FA[m]}:** {EXPLAIN[m]}")

    fig = go.Figure()
    for m in METHODS:
        tr = np.array(h["methods"][m]["trace"])
        if len(tr):
            x = h["methods"][m]["start_cost"] + tr[:, 0]
            fig.add_trace(go.Scatter(x=x, y=tr[:, 1] / h["f_best"], mode="lines+markers", name=METHOD_LABELS_FA[m],
                                     line=dict(color=COLORS[m], width=2.5)))
    fig.add_hline(y=0.999, line_dash="dot", annotation_text="هدف ۹۹٫۹٪")
    fig.update_layout(height=380, xaxis_title="هزینه (تعداد اجرای مدار)", yaxis_title="بهترین F تا اینجا / F_best",
                      xaxis_type="log", margin=dict(l=10, r=10, t=40, b=110), legend=dict(orientation="h", y=-0.38, x=0),
                      title="مسابقهٔ این رویداد")
    c2.plotly_chart(fig, width="stretch")

    st.subheader("هزینهٔ جمع‌شده در طول جریان")
    cum = w.cumulative_cost()
    fig2 = go.Figure()
    x = np.arange(1, len(w.history) + 1)
    for m in METHODS:
        fig2.add_trace(go.Scatter(x=x, y=cum[m], mode="lines+markers", name=METHOD_LABELS_FA[m],
                                  line=dict(color=COLORS[m], width=2.5)))
    fig2.update_layout(height=380, xaxis_title="شمارهٔ رویداد", yaxis_title="هزینهٔ کل تا اینجا",
                       margin=dict(l=10, r=10, t=30, b=110), legend=dict(orientation="h", y=-0.38, x=0))
    st.plotly_chart(fig2, width="stretch")
    order = sorted(METHODS, key=lambda m: cum[m][-1])
    st.markdown(f"**تا اینجا ارزان‌ترین:** {METHOD_LABELS_FA[order[0]]} ({cum[order[0]][-1]}) · "
                f"گران‌ترین: {METHOD_LABELS_FA[order[-1]]} ({cum[order[-1]][-1]}). "
                "در آزمایش E12 با ۳۵۷ رویداد، کش برای p=2 حدود ۱٫۷ برابر ارزان‌تر از زاویهٔ ثابت بود. "
                "اینجا تعداد رویداد کم است، پس ترتیب ممکن است فرق کند.")
