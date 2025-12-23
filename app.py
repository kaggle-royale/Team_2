import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import plotly.graph_objects as go

# Set page config FIRST - before any other Streamlit command
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide"
)


# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    return True


download_nltk_data()


# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('spam_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please run the training script first.")
        return None, None


# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
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


# Prediction function
def predict_spam(text, model, vectorizer):
    if not text.strip():
        return None, None

    # Preprocess
    processed_text = preprocess_text(text)
    # Vectorize
    vectorized_text = vectorizer.transform([processed_text])
    # Predict
    prediction = model.predict(vectorized_text)[0]
    # Get probability
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(vectorized_text)[0]
        confidence = max(probability) * 100
    else:
        confidence = None

    return prediction, confidence


# Streamlit App
def main():
    # Header
    st.title("📧 Email Spam Classifier")
    # st.markdown("### Detect whether an email is spam or legitimate")

    # Load model
    model, vectorizer = load_model()

    if model is None or vectorizer is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This app uses Machine Learning to classify emails as **Spam** or **Ham** (legitimate).

        **How to use:**
        1. Enter or paste email text
        2. Click 'Classify Email'
        3. View the prediction results
        """)

        st.header("Model Info")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Features:** {vectorizer.max_features}")

        st.header("Examples")
        if st.button("Load Spam Example"):
            st.session_state.example_text = "URGENT! You've won $1,000,000! Click here to claim your prize now! Limited time offer!"
        if st.button("Load Ham Example"):
            st.session_state.example_text = "Hi Rahul, I hope this email finds you well. I wanted to follow up on our meeting yesterday regarding the project timeline."

    # Main content
    col1, col2 = st.columns([2, 2])

    with col1:
        st.subheader("Enter Email Text")

        # Text input
        default_text = st.session_state.get('example_text', '')
        email_text = st.text_area(
            "Email content:",
            value=default_text,
            height=200,
            placeholder="Paste or type the email content here..."
        )

        # Classify button
        if st.button("🔍 Classify Email", type="secondary", use_container_width=True):
            if email_text.strip():
                with st.spinner("Analyzing email..."):
                    prediction, confidence = predict_spam(email_text, model, vectorizer)

                    if prediction is not None:
                        st.session_state.prediction = prediction
                        st.session_state.confidence = confidence
                        st.session_state.email_text = email_text
            else:
                st.warning("Please enter some text to classify.")

    with col2:
        st.subheader("Prediction Result")

        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence

            # Display result with color
            if prediction == 1:
                st.error("### 🚫 SPAM")
                st.markdown("This email is likely **spam**.")
                result_color = "#ff4b4b"
            else:
                st.success("### ✅ HAM (Legitimate)")
                st.markdown("This email appears to be **legitimate**.")
                result_color = "#00cc00"

            # Confidence meter
            if confidence is not None:
                st.metric("Confidence", f"{confidence:.2f}%")

                # Progress bar visualization
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': result_color},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 75], 'color': "gray"},
                            {'range': [75, 100], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            # Text statistics
            st.subheader("Text Statistics")
            word_count = len(email_text.split())
            char_count = len(email_text)

            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric("Words", word_count)
            with stat_col2:
                st.metric("Characters", char_count)

    # Batch classification
    st.markdown("---")
    st.subheader("📊 Batch Classification")

    uploaded_file = st.file_uploader("Upload a CSV file with emails", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Assume the column name is 'text' or 'email' or similar
            text_column = st.selectbox("Select the text column:", df.columns)

            if st.button("Classify All Emails"):
                with st.spinner("Classifying emails..."):
                    predictions = []
                    confidences = []

                    for text in df[text_column]:
                        pred, conf = predict_spam(str(text), model, vectorizer)
                        predictions.append("Spam" if pred == 1 else "Ham")
                        confidences.append(conf if conf else 0)

                    df['Prediction'] = predictions
                    df['Confidence (%)'] = [f"{c:.2f}" for c in confidences]

                    st.success("Classification complete!")
                    st.dataframe(df)

                    # Download results
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results",
                        csv,
                        "spam_classification_results.csv",
                        "text/csv",
                        key='download-csv'
                    )

                    # Statistics
                    spam_count = df['Prediction'].value_counts().get('Spam', 0)
                    ham_count = df['Prediction'].value_counts().get('Ham', 0)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Emails", len(df))
                    with col2:
                        st.metric("Spam Detected", spam_count)
                    with col3:
                        st.metric("Ham (Legitimate)", ham_count)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Machine Learning Spam Classifier</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()