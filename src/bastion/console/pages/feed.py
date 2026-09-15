"""Live decisions: counts, rates, the day's review budget and the newest decisions."""

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from bastion.console.common import DISCLAIMER, model_info, rate, scoring_url, store
from bastion.console.views import feed_rows, feed_summary

st.title("Live decisions")
st.caption(DISCLAIMER)


@st.fragment(run_every="5s")
def live() -> None:
    decisions = store()
    latest = decisions.latest(50)
    # The simulator replays event time, so "today" is the event day of the newest decision.
    today = latest[0].event_ts.date() if latest else datetime.now(UTC).date()
    info = model_info(scoring_url())
    budget = int(info["policy"]["reviews_per_day"]) if info else None
    summary = feed_summary(decisions, today=today, review_budget=budget)
    columns = st.columns(4)
    columns[0].metric("Decisions", f"{summary.decisions:,}")
    columns[1].metric("Block rate", rate(summary.block_rate))
    columns[2].metric("Review rate", rate(summary.review_rate))
    used = f"{summary.reviews_today:,}" + ("" if budget is None else f" of {budget:,}")
    columns[3].metric(f"Reviews on {today:%Y-%m-%d}", used)
    if latest:
        st.dataframe(pd.DataFrame(feed_rows(latest)), hide_index=True)
    else:
        st.info("No decisions yet. Start the simulator, or score a transaction on /v1/score.")


live()
