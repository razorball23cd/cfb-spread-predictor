import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# =========================
# Load / Train Model
# =========================
# For now we’ll just simulate a trained model with fake data.
# Later we’ll replace this with the real 2023 + 2024 dataset.

def train_dummy_model():
    np.random.seed(42)
    X = np.random.randn(500, 4)  # SP+, ELO features
    y = np.random.randint(0, 2, 500)  # 0 = Home fails to cover, 1 = Home covers
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(X, y)
    return model

model = train_dummy_model()

# =========================
# Prediction Function
# =========================
def predict_game(spread, sp_home, sp_away, elo_home, elo_away):
    X = np.array([[spread, sp_home - sp_away, elo_home, elo_away]])
    prob = model.predict_proba(X)[0][1]  # Probability home covers
    prediction = "✅ Home Covers" if prob > 0.5 else "❌ Home Does Not Cover"
    return prediction, float(prob)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CFB Spread Predictor", page_icon="🏈", layout="centered")
st.title("🏈 College Football Spread Predictor")

st.markdown("Enter game stats below to predict if the **home team covers the spread**.")

# --- Single Game Prediction ---
st.header("🔮 Single Game Prediction")
spread = st.number_input("Point Spread (negative = home favored)", -40.0, 40.0, -3.5, 0.5)
sp_home = st.number_input("Home SP+ Rating", 0.0, 100.0, 28.0, 0.1)
sp_away = st.number_input("Away SP+ Rating", 0.0, 100.0, 25.0, 0.1)
elo_home = st.number_input("Home ELO Rating", 500, 2500, 1600, 10)
elo_away = st.number_input("Away ELO Rating", 500, 2500, 1550, 10)

if st.button("Predict Game"):
    result, prob = predict_game(spread, sp_home, sp_away, elo_home, elo_away)
    st.subheader(result)
    st.write(f"Confidence: **{prob:.1%}**")

# --- Batch Predictions ---
st.header("📂 Batch Predictions (Upload CSV)")
st.markdown("Upload a CSV with columns: `spread,sp_plus_home,sp_plus_away,elo_home,elo_away`")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    preds, probs = [], []
    for _, row in df.iterrows():
        result, prob = predict_game(
            row["spread"], row["sp_plus_home"], row["sp_plus_away"],
            row["elo_home"], row["elo_away"]
        )
        preds.append(result)
        probs.append(prob)
    df["Prediction"] = preds
    df["Confidence"] = [f"{p:.1%}" for p in probs]
    st.dataframe(df)

    # Option to download results
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
