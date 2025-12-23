import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 70)
print("EMAIL SPAM CLASSIFIER - COMPREHENSIVE MODEL EVALUATION")
print("=" * 70)

# Download NLTK data
nltk.download('stopwords', quiet=True)

# Load the trained model and vectorizer
print("\n[1/6] Loading model and vectorizer...")
try:
    with open('spam_classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"✓ Model loaded: {type(model).__name__}")
    print(f"✓ Vectorizer loaded: TfidfVectorizer with {vectorizer.max_features} features")
except FileNotFoundError:
    print("✗ Error: Model files not found. Please run train_model.py first.")
    exit()

# Load and preprocess the dataset
print("\n[2/6] Loading and preprocessing dataset...")
df = pd.read_csv('data/email_dataset.csv', encoding='latin-1')

# Detect columns (same logic as training script)
text_col = None
label_col = None

text_variants = ['text', 'message', 'email', 'email text', 'email_text',
                 'content', 'body', 'v2', 'Message', 'Text', 'Email Text']
label_variants = ['label', 'target', 'class', 'category', 'spam',
                  'v1', 'Label', 'Target', 'Category']

for col in df.columns:
    if col in text_variants or col.lower() in [v.lower() for v in text_variants]:
        text_col = col
        break

if text_col is None:
    max_len = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > max_len:
                max_len = avg_len
                text_col = col

for col in df.columns:
    if col in label_variants or col.lower() in [v.lower() for v in label_variants]:
        label_col = col
        break

if label_col is None:
    for col in df.columns:
        if col != text_col and df[col].nunique() == 2:
            label_col = col
            break

df_clean = pd.DataFrame()
df_clean['text'] = df[text_col]
df_clean['label'] = df[label_col]
df = df_clean.drop_duplicates()

# Map labels
label_map = {}
unique_labels = df['label'].unique()
for label in unique_labels:
    label_lower = str(label).lower()
    if label_lower in ['spam', '1', '1.0', 'true']:
        label_map[label] = 1
    elif label_lower in ['ham', 'not spam', '0', '0.0', 'false']:
        label_map[label] = 0

if len(label_map) < len(unique_labels):
    for i, label in enumerate(unique_labels):
        if label not in label_map:
            label_map[label] = i

df['label'] = df['label'].map(label_map)


# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    ps = PorterStemmer()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)


df['processed_text'] = df['text'].apply(preprocess_text)
df = df[df['processed_text'].str.len() > 0]

X = df['processed_text']
y = df['label']

print(f"✓ Dataset loaded: {len(df)} samples")
print(f"✓ Class distribution: {y.value_counts().to_dict()}")

# Vectorize the data
print("\n[3/6] Vectorizing text data...")
X_vectorized = vectorizer.transform(X)
print(f"✓ Feature matrix shape: {X_vectorized.shape}")

# Make predictions
print("\n[4/6] Generating predictions...")
y_pred = model.predict(X_vectorized)

# Get prediction probabilities (if available)
if hasattr(model, 'predict_proba'):
    y_pred_proba = model.predict_proba(X_vectorized)
    y_pred_proba_positive = y_pred_proba[:, 1]
else:
    y_pred_proba_positive = None
    print("Note: This model doesn't support probability predictions")

print("✓ Predictions generated")

# Calculate metrics
print("\n[5/6] Calculating performance metrics...")
print("\n" + "=" * 70)
print("BASIC METRICS")
print("=" * 70)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)
kappa = cohen_kappa_score(y, y_pred)

print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision:          {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:             {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score:           {f1:.4f}")
print(f"Matthews Corr Coef: {mcc:.4f}")
print(f"Cohen's Kappa:      {kappa:.4f}")

# Confusion Matrix
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n                Predicted")
print(f"                Ham    Spam")
print(f"Actual  Ham     {tn:5d}  {fp:5d}")
print(f"        Spam    {fn:5d}  {tp:5d}")

print(f"\nTrue Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")

# Additional metrics
print("\n" + "=" * 70)
print("ADDITIONAL METRICS")
print("=" * 70)

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"Specificity (TNR):        {specificity:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")
print(f"False Positive Rate:      {fpr:.4f}")
print(f"False Negative Rate:      {fnr:.4f}")

# Classification Report
print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y, y_pred, target_names=['Ham (0)', 'Spam (1)'], digits=4))

# Cross-validation
print("\n" + "=" * 70)
print("CROSS-VALIDATION SCORES (5-Fold)")
print("=" * 70)
print("Performing 5-fold cross-validation...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_vectorized, y, cv=skf, scoring='accuracy')

print(f"\nFold Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Min CV Accuracy:  {cv_scores.min():.4f}")
print(f"Max CV Accuracy:  {cv_scores.max():.4f}")

# Create visualizations
print("\n[6/6] Generating visualizations...")

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Confusion Matrix Heatmap
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# 2. Normalized Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True,
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# 3. Metrics Comparison
ax3 = plt.subplot(2, 3, 3)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
metrics_values = [accuracy, precision, recall, f1, specificity]
bars = plt.bar(metrics_names, metrics_values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])
plt.ylim(0, 1.1)
plt.title('Performance Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'{metrics_values[i]:.3f}', ha='center', va='bottom', fontweight='bold')

# 4. ROC Curve
if y_pred_proba_positive is not None:
    ax4 = plt.subplot(2, 3, 4)
    fpr_roc, tpr_roc, _ = roc_curve(y, y_pred_proba_positive)
    roc_auc = auc(fpr_roc, tpr_roc)
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    print(f"✓ ROC AUC Score: {roc_auc:.4f}")

# 5. Precision-Recall Curve
if y_pred_proba_positive is not None:
    ax5 = plt.subplot(2, 3, 5)
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba_positive)
    pr_auc = auc(recall_curve, precision_curve)
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    print(f"✓ PR AUC Score: {pr_auc:.4f}")

# 6. Cross-Validation Scores
ax6 = plt.subplot(2, 3, 6)
fold_numbers = [f'Fold {i + 1}' for i in range(len(cv_scores))]
bars = plt.bar(fold_numbers, cv_scores, color='#3498db', alpha=0.7)
plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {cv_scores.mean():.4f}')
plt.ylim(0, 1.1)
plt.title('Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.legend()
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{cv_scores[i]:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_evaluation_report.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'model_evaluation_report.png'")

plt.show()

# Error Analysis
print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

# False Positives (Ham classified as Spam)
false_positives = df[(y == 0) & (y_pred == 1)]
print(f"\nFalse Positives (Ham predicted as Spam): {len(false_positives)}")
if len(false_positives) > 0:
    print("\nSample False Positives:")
    for idx, row in false_positives.head(3).iterrows():
        print(f"\n  Text: {row['text'][:100]}...")

# False Negatives (Spam classified as Ham)
false_negatives = df[(y == 1) & (y_pred == 0)]
print(f"\nFalse Negatives (Spam predicted as Ham): {len(false_negatives)}")
if len(false_negatives) > 0:
    print("\nSample False Negatives:")
    for idx, row in false_negatives.head(3).iterrows():
        print(f"\n  Text: {row['text'][:100]}...")

print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
print("\nSummary:")
print(f"  • Model Accuracy: {accuracy * 100:.2f}%")
print(f"  • Spam Detection Rate (Recall): {recall * 100:.2f}%")
print(f"  • False Positive Rate: {fpr * 100:.2f}%")
print(f"  • Cross-Validation Mean: {cv_scores.mean() * 100:.2f}%")
print(f"\n  • Report saved: model_evaluation_report.png")