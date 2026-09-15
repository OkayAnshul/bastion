"""Review queue: reviews without a verdict, largest expected fraud loss first."""

import pandas as pd
import streamlit as st

from bastion.console.common import CASE_KEY, CASE_PAGE, DISCLAIMER, store
from bastion.console.views import queue_rows

st.title("Review queue")
st.caption(
    "Reviews without a verdict, largest expected fraud loss (probability times amount) first. "
    + DISCLAIMER
)
items = store().review_queue(200)
if not items:
    st.info("The queue is empty.")
else:
    event = st.dataframe(
        pd.DataFrame(queue_rows(items)),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="queue",
    )
    if event.selection.rows:
        st.session_state[CASE_KEY] = items[event.selection.rows[0]].txn_id
        st.switch_page(CASE_PAGE)
