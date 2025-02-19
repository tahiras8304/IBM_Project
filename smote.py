import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv(r'C:\Users\PAVANI\OneDrive\Desktop\IBM Project\dataset\health prescription data.csv')  # Replace with actual dataset path

# Split the dataset into features (X) and target (y)
X = df['TEXT']
y = df['DIAGNOSIS']

# Step 1: Preprocess the text data (TF-IDF Vectorization)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Step 2: Apply SMOTE for class imbalance (if needed)
# We can modify SMOTE parameters or handle small classes
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)

# Ensure SMOTE works by filtering out classes with fewer than 2 samples
class_counts = y.value_counts()
small_classes = class_counts[class_counts < 2].index
y_filtered = y[~y.isin(small_classes)]
X_filtered = X_tfidf[~y.isin(small_classes)]

X_resampled, y_resampled = smote.fit_resample(X_filtered, y_filtered)

# Step 3: Set up StratifiedKFold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store model evaluation results
all_reports = []
all_confusion_matrices = []

# Step 4: Train the model using RandomForest with class weight adjustment
for train_idx, val_idx in cv.split(X_resampled, y_resampled):
    X_train, X_val = X_resampled[train_idx], X_resampled[val_idx]
    y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]
    
    # Train RandomForest with balanced class weights
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Store classification report as a dictionary
    report = classification_report(y_val, y_pred, output_dict=True)
    all_reports.append(report)
    
    # Store confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    all_confusion_matrices.append(cm)

    # Print the classification report for this fold
    print(classification_report(y_val, y_pred))

# Step 5: Analyze the results

# Average metrics across folds by extracting the 'macro avg' and 'weighted avg'
average_report = {
    'precision_macro_avg': np.mean([report['macro avg']['precision'] for report in all_reports]),
    'recall_macro_avg': np.mean([report['macro avg']['recall'] for report in all_reports]),
    'f1-score_macro_avg': np.mean([report['macro avg']['f1-score'] for report in all_reports]),
    'precision_weighted_avg': np.mean([report['weighted avg']['precision'] for report in all_reports]),
    'recall_weighted_avg': np.mean([report['weighted avg']['recall'] for report in all_reports]),
    'f1-score_weighted_avg': np.mean([report['weighted avg']['f1-score'] for report in all_reports]),
}

print("Average Classification Report Across Folds:")
print(pd.DataFrame(average_report, index=[0]))

# Step 6: Calculate the average confusion matrix across all folds
average_cm = np.mean(all_confusion_matrices, axis=0)

#Confusion matrix
plt.figure(figsize=(20, 15))  # Increase size
sns.heatmap(average_cm, annot=False, fmt='.2f', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_, cbar_kws={'shrink': 0.5})  # Adjust color bar
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(rotation=90)  # Rotate labels for readability
plt.yticks(rotation=0)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Average Confusion Matrix Across Folds')
plt.show()

