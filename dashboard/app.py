import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- LOAD MODELS ----------------
linear = pickle.load(open("linear.pkl", "rb"))
ridge = pickle.load(open("ridge.pkl", "rb"))
lasso = pickle.load(open("lasso.pkl", "rb"))
elastic = pickle.load(open("elastic.pkl", "rb"))
poly = pickle.load(open("poly.pkl", "rb"))
ridge_poly = pickle.load(open("ridge_poly.pkl", "rb"))

# ---------------- LOAD DATA ----------------
df = pd.read_csv("D:\energy_efficiency_data\energy_efficiency_data.csv")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Energy Predictor", layout="wide")

# ---------------- HEADER ----------------
st.markdown("# 🏠 Energy Efficiency Predictor")
st.markdown("### Predict Heating Load using Machine Learning")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔧 Input Features")

rc = st.sidebar.number_input("Relative Compactness", 0.5, 1.0, 0.8)
sa = st.sidebar.number_input("Surface Area", 500.0, 900.0, 700.0)
wa = st.sidebar.number_input("Wall Area", 200.0, 500.0, 300.0)
ra = st.sidebar.number_input("Roof Area", 100.0, 300.0, 150.0)
oh = st.sidebar.number_input("Overall Height", 3.0, 10.0, 5.0)
ori = st.sidebar.selectbox("Orientation", [2,3,4,5])
ga = st.sidebar.number_input("Glazing Area", 0.0, 0.5, 0.2)
gad = st.sidebar.selectbox("Glazing Distribution", [0,1,2,3,4,5])

input_data = np.array([[rc, sa, wa, ra, oh, ori, ga, gad]])

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediction",
    "📊 Model Comparison",
    "📌 Feature Importance",
    "📉 Model Performance"
])

# ================= TAB 1 =================
with tab1:

    st.subheader("🔮 Predict Heating Load")

    model_choice = st.selectbox(
        "Select Model",
        ["Polynomial (Best)", "Linear", "Ridge", "Lasso", "ElasticNet", "Ridge + Polynomial"]
    )

    if st.button("Predict"):

        if model_choice == "Linear":
            pred = linear.predict(input_data)
        elif model_choice == "Ridge":
            pred = ridge.predict(input_data)
        elif model_choice == "Lasso":
            pred = lasso.predict(input_data)
        elif model_choice == "ElasticNet":
            pred = elastic.predict(input_data)
        elif model_choice == "Ridge + Polynomial":
            pred = ridge_poly.predict(input_data)
        else:
            pred = poly.predict(input_data)

        st.metric("🔥 Predicted Heating Load", f"{pred[0]:.2f}")

# ================= TAB 2 =================
with tab2:

    st.subheader("📊 Compare All Models")

    results = {
        "Linear": linear.predict(input_data)[0],
        "Ridge": ridge.predict(input_data)[0],
        "Lasso": lasso.predict(input_data)[0],
        "ElasticNet": elastic.predict(input_data)[0],
        "Polynomial": poly.predict(input_data)[0],
        "Ridge+Poly": ridge_poly.predict(input_data)[0]
    }

    df_results = pd.DataFrame(results.items(), columns=["Model", "Prediction"])

    st.dataframe(df_results)

    st.bar_chart(df_results.set_index("Model"))

# ================= TAB 3 =================
with tab3:

    st.subheader("📌 Feature Importance (Polynomial Model)")

    original_features = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Distribution"
    ]

    poly_features = poly.named_steps["poly"].get_feature_names_out(original_features)
    coef = poly.named_steps["model"].coef_

    importance = pd.Series(coef, index=poly_features)
    importance = importance.sort_values(key=abs, ascending=False)[:10]

    st.bar_chart(importance)

# ================= TAB 4 =================
with tab4:

    st.subheader("📉 Model Performance (Actual vs Predicted)")

    X = df.drop("Heating_Load", axis=1)
    y = df["Heating_Load"]

    y_pred = poly.predict(X)

    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)

    # perfect line
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    st.pyplot(fig)

# ---------------- FOOTER ----------------
st.markdown("---")
st.info("Built using Machine Learning & Streamlit | Energy Efficiency Prediction Project")