"""Bastion analyst console (Phase 4). Run it with ``bastion console`` (Streamlit, port 3000).

Four pages over the decision store, one file each in ``pages/``: live decisions, the review queue
sorted by expected fraud loss, a case page, and the model and policy in service. What they show is
computed and unit-tested in ``bastion.console.views``; the page files only lay it out.
"""

import streamlit as st

st.set_page_config(page_title="Bastion analyst console", layout="wide")
st.navigation(
    [
        st.Page("pages/feed.py", title="Live decisions", default=True),
        st.Page("pages/queue.py", title="Review queue"),
        st.Page("pages/case.py", title="Case"),
        st.Page("pages/model.py", title="Model and policy"),
    ]
).run()
