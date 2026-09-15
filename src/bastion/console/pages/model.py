"""Model and policy page: what the scoring service is running, and under which cost assumptions."""

import pandas as pd
import streamlit as st

from bastion.console.common import model_info, scoring_url

st.title("Model and policy")
url = st.text_input("Scoring service", value=scoring_url())
st.session_state["scoring_url"] = url
info = model_info(url)
if info is None:
    st.warning(f"No answer from {url}/v1/model.")
else:
    policy, costs = info["policy"], info["policy"]["costs"]
    columns = st.columns(3)
    columns[0].metric("Model", info["model_version"])
    columns[1].metric("Reviews per day", f"{policy['reviews_per_day']:,}")
    columns[2].metric(
        "Review threshold",
        f"{policy['review_threshold']:,.2f} {costs['currency']}",
        help="A review must be expected to save more than this over approving or blocking.",
    )
    if policy["tuned"]:
        st.write("The threshold was tuned by `bastion policy sweep` for this model and budget.")
    else:
        st.warning("Untuned threshold (0). Run `bastion policy sweep` and set BASTION_POLICY_PATH.")
    st.subheader("Assumed costs")
    st.caption("configs/costs.yaml. These are assumptions, not measurements.")
    table = pd.DataFrame(list(costs.items()), columns=["Assumption", "Value"])
    st.table(table.set_index("Assumption"))
    st.subheader("Service")
    st.json(
        {
            "spec_fingerprint": info["spec_fingerprint"],
            "inputs": info["inputs"],
            "decision_log": info["decision_log"],
            "store_errors": info["store_errors"],
        }
    )
