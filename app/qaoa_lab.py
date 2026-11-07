"""Interactive demo of the project: QAOA warm starts for dynamic vehicle routing.

Run from the repository root:
    pip install -e ".[app]"
    streamlit run app/qaoa_lab.py
Every number on these pages is computed live by the project code in src/qws (no precomputed results,
except the E14 reference curve on the theory page, which is labelled as such).
"""
import streamlit as st

st.set_page_config(page_title="آزمایشگاه QAOA", page_icon="🚚", layout="wide")

from views import clusters, database, landscape, stream, theory  # noqa: E402

pages = [
    st.Page(clusters.render, title="۱. سفارش‌ها و دو خوشه", default=True),
    st.Page(landscape.render, title="۲. پیچ‌های QAOA", url_path="landscape"),
    st.Page(stream.render, title="۳. سفارش جدید و شروع گرم", url_path="stream"),
    st.Page(database.render, title="۴. کش در پایگاه داده", url_path="database"),
    st.Page(theory.render, title="۵. نظریه: جابه‌جایی زاویه", url_path="theory"),
]
st.navigation(pages).run()
