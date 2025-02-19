import pandas as pd
# Load the dataset
data = pd.read_csv(r"C:\Users\PAVANI\OneDrive\Desktop\IBM Project\dataset\health prescription data.csv")
# Check the first few rows of the dataset to understand its structure
print(data.head())

import re
# Extract dates from the 'TEXT' column using a regular expression
data['extracted_date'] = data['TEXT'].apply(lambda x: re.search(r'\d{4}-\d{1,2}-\d{1,2}', x).group(0) if re.search(r'\d{4}-\d{1,2}-\d{1,2}', x) else None)
# Display the extracted dates
print(data[['TEXT', 'extracted_date']].head())
# Normalize the extracted date to the format 'yyyy-mm-dd'
data['normalized_date'] = pd.to_datetime(data['extracted_date'], errors='coerce').dt.strftime('%Y-%m-%d')
# Display the first few rows of the updated dataset
print(data[['extracted_date', 'normalized_date']].head())

import re
from nltk.corpus import stopwords
# Download stopwords if not already done
import nltk
nltk.download('stopwords')
# Preprocess the TEXT column
def preprocess_text(text):
    # Remove dates in the format [**YYYY-MM-DD**] and any special characters
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words]) 
    return text
# Apply preprocessing to the TEXT column
data['processed_text'] = data['TEXT'].apply(preprocess_text)
# Display the first few rows of the processed text
print(data[['TEXT', 'processed_text']].head())

from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
# Fit and transform the processed text data
X = vectorizer.fit_transform(data['processed_text'])
# Display the shape of the transformed data
print(X.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# Assuming 'DIAGNOSIS' is the target variable
y = data['DIAGNOSIS']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
# Train the model
model.fit(X_train, y_train)
# Make predictions on the test data
y_pred = model.predict(X_test)
# Print the classification report
print(classification_report(y_test, y_pred, zero_division=1))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Assuming 'processed_text' is the text column and 'DIAGNOSIS' is the target column
X = data['processed_text']
y = data['DIAGNOSIS']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# Initialize the Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
rf_model.fit(X_train_tfidf, y_train)
# Make predictions on the test set
y_pred = rf_model.predict(X_test_tfidf)
# Print the classification report
print(classification_report(y_test, y_pred, zero_division=1))




