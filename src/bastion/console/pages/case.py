"""Case page: facts, the decision and its expected costs, reason codes, card context, verdict."""

import altair as alt
import pandas as pd
import streamlit as st

from bastion.console.common import CASE_KEY, DISCLAIMER, store
from bastion.console.views import case_view, feed_rows
from bastion.plotting import SERIES


def reason_chart(reasons: list[dict[str, object]]) -> alt.Chart:
    chart: alt.Chart = (
        alt.Chart(pd.DataFrame(reasons))
        .mark_bar(color=SERIES[0], cornerRadiusEnd=4)
        .encode(
            x=alt.X("contribution:Q", title="Contribution to the model's log-odds"),
            y=alt.Y("reason:N", sort="-x", title=None),
            tooltip=["reason", "value", alt.Tooltip("contribution:Q", format=".3f")],
        )
    )
    return chart


st.title("Case")
decisions = store()
txn_id = st.text_input("Transaction id", value=st.session_state.get(CASE_KEY, "")).strip()
view = case_view(decisions, txn_id) if txn_id else None
if not txn_id:
    st.info("Pick a case from the review queue, or enter a transaction id.")
elif view is None:
    st.warning(f"No decision recorded for {txn_id}.")
else:
    st.session_state[CASE_KEY] = txn_id
    st.caption(DISCLAIMER)

    left, right = st.columns(2)
    with left:
        st.subheader("Facts")
        st.table(pd.DataFrame(view.facts, columns=["Field", "Value"]).set_index("Field"))
        if view.expected_costs:
            st.subheader("Expected cost of each action")
            st.caption("Under the assumed costs in configs/costs.yaml; cheapest first.")
            costs = pd.DataFrame(view.expected_costs, columns=["Action", "Expected cost"])
            st.table(costs.set_index("Action"))
    with right:
        st.subheader("What the model leaned on")
        if view.reasons:
            rows: list[dict[str, object]] = [
                {
                    "reason": code.description,
                    "value": "n/a" if code.value is None else str(code.value),
                    "contribution": code.contribution,
                }
                for code in view.reasons
            ]
            st.altair_chart(reason_chart(rows))
            st.caption(
                "TreeSHAP contributions for this transaction: what raised the model's score, not"
                " what caused the fraud. The decision also depends on the amount and the costs."
            )
        else:
            st.caption("Reason codes are attached to reviewed and blocked transactions.")

    if view.baseline:
        st.subheader("Against the card's own recent behaviour")
        columns = ["Measure", "This transaction", "The card before it"]
        st.table(pd.DataFrame(view.baseline, columns=columns).set_index("Measure"))
    st.subheader("Card history")
    if view.history:
        st.dataframe(pd.DataFrame(feed_rows(view.history)), hide_index=True)
    else:
        st.caption("No earlier decisions for this card.")

    st.subheader("Analyst verdict")
    if view.verdict is not None:
        verdict = view.verdict
        st.success(
            f"{verdict.verdict} · {verdict.analyst} · {verdict.decided_at}"
            + (f" · {verdict.note}" if verdict.note else "")
        )
    with st.form("verdict"):
        choice = st.radio("Verdict", ["fraud", "legitimate"], horizontal=True)
        analyst = st.text_input("Analyst", value=st.session_state.get("analyst", ""))
        note = st.text_area("Note")
        if st.form_submit_button("Save verdict"):
            if not analyst.strip():
                st.error("Enter the analyst's name.")
            else:
                decisions.record_verdict(
                    txn_id,
                    "fraud" if choice == "fraud" else "legitimate",
                    analyst=analyst.strip(),
                    note=note,
                )
                st.session_state["analyst"] = analyst.strip()
                st.rerun()
