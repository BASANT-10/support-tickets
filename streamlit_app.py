# STEP 0: Install necessary packages
!pip install -q pandas matplotlib
# STEP 1: Imports
import pandas as pd
import re, io
from google.colab import files
import matplotlib.pyplot as plt
# STEP 2: Define Tactic Dictionary
tactics = {
    "urgency_marketing": {'now', 'today', 'limited', 'hurry', 'exclusive'},
    "social_proof": {'bestseller', 'popular', 'trending', 'recommended'},
    "discount_marketing": {'sale', 'discount', 'deal', 'free', 'offer'}
}
# STEP 3: Helper Functions
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
def classify_and_score(text, tactic_keywords):
    words = text.split()
    matched = [kw for kw in tactic_keywords if kw in words]
    predicted = 1 if matched else 0
    score = len(matched) / len(tactic_keywords)
    return predicted, score, matched
# STEP 4: Upload CSV File
print("📁 Upload your CSV with at least 'ID' and 'Statement' columns:")
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(next(iter(uploaded.values()))))
# STEP 5: Choose Tactic
print("\n🎯 Choose a tactic:")
for i, t in enumerate(tactics): print(f"{i}. {t}")
choice = int(input("Enter tactic number: "))
selected_tactic = list(tactics)[choice]
print(f"✅ Selected tactic: {selected_tactic}")
# STEP 6: Clean Text
df['clean'] = df['Statement'].apply(clean_text)
# STEP 7: Classify + Score
results = df['clean'].apply(lambda x: classify_and_score(x, tactics[selected_tactic]))
df['predicted'] = results.apply(lambda x: x[0])         # Binary prediction (0/1)
df['score'] = results.apply(lambda x: x[1])             # Continuous score
df['matched_keywords'] = results.apply(lambda x: ', '.join(x[2]))  # Optional for inspection
df['category'] = df['predicted'].apply(lambda x: selected_tactic if x == 1 else 'uncategorized')
# STEP 8: Preview Results
print("\n🔍 Preview:")
print(df[['ID', 'Statement', 'category', 'score', 'matched_keywords']].head())
# STEP 9: Save Output
df.to_csv("classified_with_scores.csv", index=False)
print("\n📥 Download your results:")
files.download("classified_with_scores.csv")
# STEP 10 (Optional): Correlation if likes/comments present
if 'likes' in df.columns and 'comments' in df.columns:
    print("\n📊 Correlation:")
    print("Score vs Likes:", df['score'].corr(df['likes']))
    print("Score vs Comments:", df['score'].corr(df['comments']))
    plt.scatter(df['score'], df['likes'], alpha=0.6)
    plt.xlabel('Score')
    plt.ylabel('Likes')
    plt.title('Score vs Likes')
    plt.grid(True)
    plt.show()
