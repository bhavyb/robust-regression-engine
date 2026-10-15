import json, joblib
import pandas as pd, streamlit as st

st.set_page_config(page_title="Robust Regression Engine", page_icon="🏠", layout="wide")

@st.cache_resource
def load():
    return joblib.load("final_model.joblib"), json.load(open("model_meta.json"))

model, meta = load()
FEATURES = meta["features"]; R = meta["feature_ranges"]

st.title("🏠 Robust Regression Engine")
st.caption(f"House price prediction  |  Best model: **{meta['best_model']}**")

tab1, tab2, tab3 = st.tabs(["Predict price", "Model comparison", "About"])

with tab1:
    st.subheader("Enter property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.slider("Area (sqft)", 500, 4500, int(R["area_sqft"][2]), 10)
        bed = st.number_input("Bedrooms", 1, 8, int(R["bedrooms"][2]))
        bath = st.number_input("Bathrooms", 1, 7, int(R["bathrooms"][2]))
    with c2:
        loc = st.slider("Location score (1-10)", 1.0, 10.0, float(R["location_score"][2]), 0.1)
        age = st.slider("Property age (years)", 0, 80, int(R["property_age"][2]))
        dist = st.slider("Distance from city (km)", 0.5, 40.0, float(R["distance_city_km"][2]), 0.1)
    with c3:
        crime = st.slider("Crime rate index", 0.0, 12.0, float(R["crime_rate_index"][2]), 0.1)
        school = st.checkbox("Near school", True)
        metro = st.checkbox("Near metro", True)
        year = st.slider("Sale year", 2010, 2030, 2023)

    row = pd.DataFrame([{"area_sqft": area, "bedrooms": bed, "bathrooms": bath, "location_score": loc,
                         "property_age": age, "distance_city_km": dist, "near_school": int(school),
                         "near_metro": int(metro), "crime_rate_index": crime, "sale_year": year}])[FEATURES]
    if st.button("Predict house price", type="primary"):
        lakh = float(model.predict(row)[0]); mae = meta["test_metrics"]["MAE"]
        st.metric("Estimated price", f"₹ {lakh*1e5:,.0f}", f"{lakh:,.2f} Lakh")
        st.info(f"Typical error of this model (MAE on unseen data) ≈ ₹ {mae*1e5:,.0f}. "
                f"Likely range: ₹ {max(lakh-mae,0)*1e5:,.0f} – ₹ {(lakh+mae)*1e5:,.0f}")
        if year > 2023: st.warning("Sale year is beyond the training data (2010–2023): treat the result as an extrapolation.")

with tab2:
    st.subheader("All models (Lakh INR)")
    st.dataframe(pd.read_csv("model_comparison.csv", index_col=0).sort_values("Test_RMSE"), use_container_width=True)
    st.subheader("Cross-validation strategies (Ridge)")
    st.dataframe(pd.read_csv("cv_strategies.csv", index_col=0), use_container_width=True)
    st.subheader("Feature importance (Random Forest)")
    st.bar_chart(pd.Series(meta["importance"]))

with tab3:
    m = meta["test_metrics"]
    st.write("Pipeline: preprocessing → regularized linear models, tree models and SVR → cross-validation → best model selected by validation RMSE.")
    a, b, c = st.columns(3)
    a.metric("Test R²", f"{m['R2']:.4f}"); b.metric("Test RMSE (Lakh)", f"{m['RMSE']:.2f}"); c.metric("Test MAE (Lakh)", f"{m['MAE']:.2f}")
