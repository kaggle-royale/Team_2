import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords', quiet=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/email_dataset.csv', encoding='latin-1')

# Display basic info
print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nDataset info:")
print(df.info())

# Automatically detect text and label columns
print("\n" + "=" * 50)
print("Detecting columns...")

text_col = None
label_col = None

# Common text column names
text_variants = ['text', 'message', 'email', 'email text', 'email_text',
                 'content', 'body', 'v2', 'Message', 'Text', 'Email Text']

# Common label column names
label_variants = ['label', 'target', 'class', 'category', 'spam',
                  'v1', 'Label', 'Target', 'Category']

# Find text column
for col in df.columns:
    if col in text_variants or col.lower() in [v.lower() for v in text_variants]:
        text_col = col
        break

# If not found, look for the column with longest average text length
if text_col is None:
    max_len = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > max_len:
                max_len = avg_len
                text_col = col

# Find label column
for col in df.columns:
    if col in label_variants or col.lower() in [v.lower() for v in label_variants]:
        label_col = col
        break

# If not found, look for column with 2 unique values
if label_col is None:
    for col in df.columns:
        if col != text_col and df[col].nunique() == 2:
            label_col = col
            break

print(f"Text column detected: {text_col}")
print(f"Label column detected: {label_col}")

if text_col is None or label_col is None:
    print("\nERROR: Could not automatically detect columns!")
    print("\nAvailable columns:", df.columns.tolist())
    print("\nPlease specify columns manually:")
    print("text_col = 'your_text_column_name'")
    print("label_col = 'your_label_column_name'")
    exit()

# Create a clean dataframe with standard column names
df_clean = pd.DataFrame()
df_clean['text'] = df[text_col]
df_clean['label'] = df[label_col]

# Remove any extra columns and duplicates
df = df_clean.drop_duplicates()

print("\nUnique labels:", df['label'].unique())

# Convert labels to binary (0: ham, 1: spam)
# Handle various label formats
label_map = {}
unique_labels = df['label'].unique()

for label in unique_labels:
    label_lower = str(label).lower()
    if label_lower in ['spam', '1', '1.0', 'true']:
        label_map[label] = 1
    elif label_lower in ['ham', 'not spam', '0', '0.0', 'false']:
        label_map[label] = 0

# If still not mapped, use the first unique value as 0, second as 1
if len(label_map) < len(unique_labels):
    for i, label in enumerate(unique_labels):
        if label not in label_map:
            label_map[label] = i

df['label'] = df['label'].map(label_map)

print(f"\nLabel mapping: {label_map}")
print(f"Class distribution:\n{df['label'].value_counts()}")


# Text preprocessing function
def preprocess_text(text):
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenization and stemming
    ps = PorterStemmer()
    words = text.split()
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    words = [ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)


# Apply preprocessing
print("\nPreprocessing text...")
df['processed_text'] = df['text'].apply(preprocess_text)

# Remove any empty texts after preprocessing
df = df[df['processed_text'].str.len() > 0]

# Split the data
X = df['processed_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Vectorization
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.8)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_vec.shape}")

# Train multiple models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}...")
    model.fit(X_train_vec, y_train)

    # Predictions
    y_pred = model.predict(X_test_vec)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print(f"\n{'=' * 50}")
print(f"Best Model: {best_model_name} with accuracy: {results[best_model_name]:.4f}")

# Save the best model and vectorizer
print("\nSaving model and vectorizer...")
with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel training complete!")
print("Files created:")
print("  - spam_classifier_model.pkl")
print("  - vectorizer.pkl")
print("\nYou can now run the Streamlit app with: streamlit run app.py")