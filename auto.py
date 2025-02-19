import pandas as pd
from fuzzywuzzy import process
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
file_path = r"C:\Users\PAVANI\OneDrive\Desktop\IBM Project\dataset\health prescription data.csv"  # Update the path if needed
df = pd.read_csv(file_path)

# 🔹 Step 1: Fix Misspellings & Abbreviations
medical_terms = {
    "HTN": "Hypertension",
    "DM": "Diabetes Mellitus",
    "COPD": "Chronic Obstructive Pulmonary Disease",
    "S/P FALL": "Status Post Fall",
    "PEDISTRIAN STRUCK": "PEDESTRIAN STRUCK"
}

def correct_spelling(text):
    if pd.isna(text):
        return None
    # Check if abbreviation exists in dictionary
    if text in medical_terms:
        return medical_terms[text]
    best_match = process.extractOne(text, medical_terms.values())
    return best_match[0] if best_match and best_match[1] > 80 else text  # Fix only if confidence > 80%

df["CLEAN_DIAGNOSIS"] = df["DIAGNOSIS"].apply(correct_spelling)

# 🔹 Step 2: Map to ICD-10 Codes (Simplified Mapping Example)
icd_mapping = {
    "Hypertension": "I10",
    "Diabetes Mellitus": "E11",
    "Chronic Obstructive Pulmonary Disease": "J44"
}

df["ICD10_CODE"] = df["CLEAN_DIAGNOSIS"].map(icd_mapping)

# 🔹 Step 3: Detect Anomalies Using Isolation Forest
diagnosis_counts = df["CLEAN_DIAGNOSIS"].value_counts().to_dict()
df["DIAGNOSIS_COUNT"] = df["CLEAN_DIAGNOSIS"].map(diagnosis_counts)

model = IsolationForest(contamination=0.05, random_state=42)
df["ANOMALY"] = model.fit_predict(df[["DIAGNOSIS_COUNT"]])

df_cleaned = df[df["ANOMALY"] == 1]  # Keep only normal cases

# 🔹 Step 4: Suggest Diagnoses for Unclear/Missing Cases
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df["TEXT"].dropna())

def suggest_diagnosis(missing_text):
    if pd.isna(missing_text):
        return None
    text_vector = vectorizer.transform([missing_text])
    similarity_scores = cosine_similarity(text_vector, X_tfidf)
    best_match_idx = similarity_scores.argmax()
    return df["CLEAN_DIAGNOSIS"].iloc[best_match_idx]

df_cleaned = df_cleaned.copy()  # Create an explicit copy
df_cleaned["SUGGESTED_DIAGNOSIS"] = df_cleaned["TEXT"].apply(suggest_diagnosis)

# 🔹 Save the cleaned dataset
df_cleaned.to_csv("cleaned_healthcare_data.csv", index=False)
df_cleaned.to_json("cleaned_healthcare_data.json", orient="records", indent=4)

print("✅ Data cleaning complete! Saved as CSV & JSON.")
