import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and vectorizer
model = joblib.load('text_classification_model.pkl')  # Load the trained model
vectorizer = joblib.load('vectorizer.pkl')  # Load the saved vectorizer

# Assuming you have your dataset 'df' with the 'TEXT' and 'DIAGNOSIS' columns
df = pd.read_csv(r'C:\Users\PAVANI\OneDrive\Desktop\IBM Project\dataset\health prescription data.csv')  # Replace with actual dataset path

# Randomly sample 5 rows from the dataset for testing
sample_data = df[['TEXT', 'DIAGNOSIS']].sample(n=2, random_state=42)

# Extract the sample texts
sample_texts = sample_data['TEXT'].tolist()

# Display the sample texts along with their true diagnoses
for i, text in enumerate(sample_texts):
    print(f"Sample Text {i+1}:")
    print(f"Text: {text}")
    print(f"True Diagnosis: {sample_data.iloc[i]['DIAGNOSIS']}")
    print("="*50)

# Preprocess the sample texts using the loaded vectorizer
sample_tfidf = vectorizer.transform(sample_texts)

# Make predictions using the loaded model
predictions = model.predict(sample_tfidf)

# Display predictions alongside the true diagnosis
for i, prediction in enumerate(predictions):
    print(f"Sample Text {i+1}:")
    print(f"True Diagnosis: {sample_data.iloc[i]['DIAGNOSIS']}")
    print(f"Predicted Diagnosis: {prediction}")
    print("="*50)
