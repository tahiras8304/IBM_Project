# NLP-Based Automatic Cleansing for Healthcare Data

# Import necessary libraries
import pandas as pd
import numpy as np
import spacy
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from textblob import TextBlob

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Step 1: Data Preparation
def load_and_clean_data(filepath):
    """Load and preprocess prescription data."""
    data = pd.read_csv(r'C:\Users\PAVANI\OneDrive\Desktop\IBM Project\dataset\health prescription data.csv')
    data.drop_duplicates(inplace=True)
    data.fillna(method='ffill', inplace=True)
    return data

# Step 2: Visualization
def visualize_data(data):
    """Generate visualizations for understanding data patterns."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data['CATEGORY'], order=data['CATEGORY'].value_counts().index[:10])
    plt.title("Top 10 Most Common Categories")
    plt.xticks(rotation=45)
    plt.show()

# Step 3: Named Entity Recognition (NER)
def extract_entities(text):
    """Extract medical entities using spaCy."""
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Step 4: Error Detection and Correction
def correct_spelling(text, word_database):
    """Correct spelling errors using fuzzy matching."""
    corrected_word = process.extractOne(text, word_database)[0]
    return corrected_word

# Step 5: Anomaly Detection
def detect_anomalies(data):
    """Detect anomalies in text length using Isolation Forest."""
    data['TEXT_LENGTH'] = data['TEXT'].apply(len)
    model = IsolationForest(contamination=0.01)
    data['Text_Anomaly'] = model.fit_predict(data[['TEXT_LENGTH']])
    return data

# Step 6: Standardization
def standardize_data(data, category_database):
    """Standardize categories."""
    data['Standardized Category'] = data['CATEGORY'].apply(lambda x: correct_spelling(x, category_database))
    return data

# Step 7: Deployment API
def deploy_model_as_api():
    """Placeholder for API deployment logic."""
    print("Deploying model as an API using Flask or FastAPI...")

# Step 8: Reporting and Visualization
def generate_report(data):
    """Generate a report for flagged anomalies."""
    anomalies = data[data['Text_Anomaly'] == -1]
    anomalies.to_csv('anomaly_report.csv', index=False)
    print("Anomaly report saved as 'anomaly_report.csv'")

# Sample Workflow
if __name__ == "__main__":
    # Step 1: Load data
    filepath = "health prescription data.csv"
    category_database = ["Emergency", "Elective", "Urgent"]  # Example category database
    data = load_and_clean_data(filepath)

    # Step 2: Visualize data
    visualize_data(data)

    # Step 3: Extract entities
    data['Entities'] = data['TEXT'].apply(extract_entities)

    # Step 4: Correct spelling
    data = standardize_data(data, category_database)

    # Step 5: Detect anomalies
    data = detect_anomalies(data)

    # Step 6: Generate report
    generate_report(data)

    # Step 7: Deploy model as API
    deploy_model_as_api()
