"""Start page. Run with:  streamlit run Home.py   (the pages/ folder adds pages 1-4 to the sidebar)."""
import streamlit as st

from lib import api
from lib.layout import hero

hero("Neighbour impact of a cell outage",
     "When a mobile cell goes down, its users move to the neighbouring cells. This app predicts each neighbour's "
     "latency, utilization and packet loss at the next 3-hourly reading, using only data from before the outage.")

# st.markdown("""
# | User Guide
# |---|---|---|
# | **1 · Predict an outage** | A real outage from the list, or a cell and a time | Predictions for every neighbour, next to the actual values; edit any input and predict again |
# | **2 · What-if** | An outage, then sliders: failed cell's load, cells already down, time of day | How the predicted impact changes |
# | **3 · Upload data** | A CSV or ZIP in the Dataset 2 format | The same predictions for your own readings |
# | **4 · API** | JSON | The endpoints, examples, and a request tester |
# """)

st.markdown("""

### **USER GUIDE**

1. Predict an outage: View the predicted impact on neighbouring cells whenever there is an outage.

2. What-If Analysis: Select an outage and use sliders to adjust the failed cell's load, the number of cells already down, and the time of day. See how these changes affect the predicted impact.

3. Upload Data: Upload a CSV or ZIP file with new data for prediction.

4. API Access: Explore the available API endpoints, view JSON request and response examples, and test requests using the built-in tester.

""")

# c1, c2 = st.columns(2)
# with c1:
#     st.subheader("How a prediction is made")
#     st.markdown("""
# 1. The app finds the failed cell's neighbours (listed in Dataset 1, or on the same site) that were up just before
#    the outage.
# 2. It takes their last three readings (now, 3 h and 6 h earlier), their reading one week earlier, and the failed
#    cell's last reading.
# 3. It builds the same 66 features as the notebook and runs the three Gradient Boosting models.
# 4. A neighbour is flagged **degraded** if a KPI is predicted to rise more than 20% above its value before the
#    outage, and **severe** above 50%.
# """)
# with c2:
#     st.subheader("Model")
#     try:
#         info = api.model_info()
#         st.dataframe([{"KPI": t, "Test MAE": round(m["MAE"], 3), "Test R²": round(m["R2_level"], 3)}
#                       for t, m in info["test_metrics"].items()], hide_index=True)
#         st.caption(f"{info['model']}, {info['n_features']} features, trained on data until "
#                    f"{str(info['trained_until'])[:10]}. scikit-learn {info['sklearn_running']} "
#                    f"(trained with {info['sklearn_trained']}).")
#     except api.ApiError as exc:
#         api.show_error(exc)
