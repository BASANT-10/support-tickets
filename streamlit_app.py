# app.py
import io
import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- STEP 1: Define tactic dictionary ----------
TACTICS = {
    "Urgency marketing": {'now', 'today', 'limited', 'hurry', 'exclusive'},
    "Social proof": {'bestseller', 'popular', 'trending', 'recommended'},
    "Discount marketing": {'sale', 'discount', 'deal', 'free', 'offer'}
}

# ---------- STEP 2: Helper functions ----------
def clean_text(text: str) -> str:
    """Lower‑case, remove punctuation & keep alphanumerics / whitespace."""
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())

def classify_and_score(text: str, keywords: set) -> tuple[int, float, list[str]]:
    words = text.split()
    matched = [kw for kw in keywords if kw in words]
    predicted = 1 if matched else 0
    score = len(matched) / len(keywords)
    return predicted, score, matched

# ---------- UI ----------
st.title("Dictionary‑based Tactic Classifier")

uploaded_file = st.file_uploader(
    "📁 Upload a CSV that contains at least 'ID' and 'Statement' columns",
    type="csv"
)

if uploaded_file:
    # STEP 3: Read data
    df = pd.read_csv(uploaded_file)

    # Sanity check
    required_cols = {'ID', 'Statement'}
    if not required_cols.issubset(df.columns):
        st.error(f"Your file must contain the columns: {', '.join(required_cols)}")
        st.stop()

    # STEP 4: Choose tactic
    tactic_name = st.selectbox("🎯 Choose a tactic", list(TACTICS.keys()))
    tactic_keywords = TACTICS[tactic_name]

    # ---------- Processing ----------
    with st.spinner("Classifying…"):
        df['clean'] = df['Statement'].apply(clean_text)
        results = df['clean'].apply(lambda x: classify_and_score(x, tactic_keywords))
        df['predicted'] = results.apply(lambda x: x[0])
        df['score'] = results.apply(lambda x: x[1])
        df['matched_keywords'] = results.apply(lambda x: ', '.join(x[2]))
        df['category'] = df['predicted'].apply(lambda x: tactic_name if x == 1 else 'uncategorized')

    # ---------- Preview ----------
    st.subheader("🔍 Preview")
    st.dataframe(df[['ID', 'Statement', 'category', 'score', 'matched_keywords']].head())

    # ---------- Download ----------
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download full results as CSV",
        data=csv_bytes,
        file_name="classified_with_scores.csv",
        mime="text/csv"
    )

    # ---------- Optional correlation ----------
    if {'likes', 'comments'}.issubset(df.columns):
        st.subheader("📊 Correlation with Engagement")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Score ↔ Likes:", round(df['score'].corr(df['likes']), 3))
            fig1, ax1 = plt.subplots()
            ax1.scatter(df['score'], df['likes'], alpha=0.6)
            ax1.set_xlabel('Score')
            ax1.set_ylabel('Likes')
            ax1.set_title('Score vs Likes')
            ax1.grid(True)
            st.pyplot(fig1)
        with col2:
            st.write("Score ↔ Comments:", round(df['score'].corr(df['comments']), 3))
            fig2, ax2 = plt.subplots()
            ax2.scatter(df['score'], df['comments'], alpha=0.6)
            ax2.set_xlabel('Score')
            ax2.set_ylabel('Comments')
            ax2.set_title('Score vs Comments')
            ax2.grid(True)
            st.pyplot(fig2)
