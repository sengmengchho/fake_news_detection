import streamlit as st
import numpy as np
import re
import pickle
from pathlib import Path
import sys
import os

# Disable GPU for model loading
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .fake-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
    }
    .true-badge {
        background-color: #51cf66;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📰 Fake News Detector")
st.markdown("---")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses an ensemble of 3 deep learning models:
    - **Dense Neural Network**
    - **CNN (Convolutional Neural Network)**
    - **BiLSTM (Bidirectional LSTM)**
    
    A meta-learner combines their predictions for optimal results.
    """)
    st.markdown("---")
    st.markdown("Built with TensorFlow & Streamlit")

# Load models (using caching to avoid reloading)
@st.cache_resource
def load_models():
    """Load trained models from files"""
    models_dir = Path(__file__).parent / "models"
    
    try:
        st.write("Loading models...")
        dense_model = load_model(str(models_dir / "dense_model.h5"))
        cnn_model = load_model(str(models_dir / "cnn_model.h5"))
        bilstm_model = load_model(str(models_dir / "bilstm_model.h5"))
        meta_model = load_model(str(models_dir / "meta_model.h5"))
        
        with open(models_dir / "tfidf_vectorizer.pkl", "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        
        st.success("✅ Models loaded successfully!")
        return dense_model, cnn_model, bilstm_model, meta_model, tfidf_vectorizer
    except Exception as e:
        st.error(f"❌ Failed to load models: {str(e)}")
        st.info(f"Looking in: {models_dir}")
        return None, None, None, None, None

@st.cache_resource
def get_nltk_stopwords():
    return stopwords.words('english')

def clean_text(text):
    """Clean and preprocess text"""
    stop_words = get_nltk_stopwords()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to string
    return " ".join(tokens)

def predict_news(text, dense_model, cnn_model, bilstm_model, meta_model, tfidf_correct):
    """Predict if news is fake or true"""
    # Clean and vectorize
    clean_sample = clean_text(text)
    sample_vec = tfidf_correct.transform([clean_sample]).astype(np.float32).toarray()
    
    # Get predictions from base models
    dense_pred = dense_model.predict(sample_vec, verbose=0)[0][0]
    cnn_pred = cnn_model.predict(sample_vec, verbose=0)[0][0]
    bilstm_pred = bilstm_model.predict(sample_vec, verbose=0)[0][0]
    
    # Stack predictions for meta-learner with correct shape (1, 3)
    meta_features = np.array([[float(dense_pred), float(cnn_pred), float(bilstm_pred)]])
    final_pred = meta_model.predict(meta_features, verbose=0)[0][0]
    
    # Convert to class labels
    dense_class = "TRUE" if dense_pred > 0.5 else "FAKE"
    cnn_class = "TRUE" if cnn_pred > 0.5 else "FAKE"
    bilstm_class = "TRUE" if bilstm_pred > 0.5 else "FAKE"
    final_class = "TRUE" if final_pred > 0.5 else "FAKE"
    
    return {
        'dense_prob': float(dense_pred),
        'cnn_prob': float(cnn_pred),
        'bilstm_prob': float(bilstm_pred),
        'final_prob': float(final_pred),
        'dense_class': dense_class,
        'cnn_class': cnn_class,
        'bilstm_class': bilstm_class,
        'final_class': final_class,
        'clean_text': clean_sample
    }

# Main UI
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter News Text")
    user_input = st.text_area(
        "Paste the news article you want to check:",
        height=200,
        placeholder="Enter your news text here..."
    )

with col2:
    st.subheader("⚙️ Options")
    show_details = st.checkbox("Show detailed analysis", value=True)
    show_probabilities = st.checkbox("Show confidence scores", value=True)

st.markdown("---")

# Prediction
if user_input.strip():
    if st.button("🔍 Analyze", use_container_width=True):
        try:
            # Load models
            dense_model, cnn_model, bilstm_model, meta_model, tfidf_correct = load_models()
            
            if all([dense_model, cnn_model, bilstm_model, meta_model, tfidf_correct]):
                with st.spinner("🔄 Analyzing..."):
                    results = predict_news(
                        user_input,
                        dense_model,
                        cnn_model,
                        bilstm_model,
                        meta_model,
                        tfidf_correct
                    )
                
                # Display main result prominently
                st.markdown("### 🎯 Final Prediction")
                
                col_pred, col_conf = st.columns([2, 1])
                with col_pred:
                    if results['final_class'] == 'FAKE':
                        st.markdown(f"<div class='fake-badge'>⚠️ FAKE NEWS</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='true-badge'>✅ TRUE NEWS</div>", unsafe_allow_html=True)
                
                with col_conf:
                    confidence = (results['final_prob'] if results['final_class'] == 'TRUE' else 1 - results['final_prob']) * 100
                    st.metric("Confidence", f"{confidence:.2f}%")
                
                # Show all probabilities in a table
                st.markdown("---")
                st.markdown("### 📊 All Model Probabilities")
                
                prob_data = {
                    'Model': ['Dense NN', 'CNN', 'BiLSTM', 'Meta-Learner'],
                    'Probability': [
                        f"{results['dense_prob']:.4f}",
                        f"{results['cnn_prob']:.4f}",
                        f"{results['bilstm_prob']:.4f}",
                        f"{results['final_prob']:.4f}"
                    ],
                    'Prediction': [
                        results['dense_class'],
                        results['cnn_class'],
                        results['bilstm_class'],
                        results['final_class']
                    ]
                }
                
                import pandas as pd
                df_probs = pd.DataFrame(prob_data)
                st.table(df_probs)
                
                # Probability chart
                if show_probabilities:
                    st.markdown("---")
                    st.markdown("### 📈 Probability Visualization")
                    
                    chart_data = pd.DataFrame({
                        'Model': ['Dense NN', 'CNN', 'BiLSTM', 'Meta-Learner'],
                        'Probability': [
                            results['dense_prob'],
                            results['cnn_prob'],
                            results['bilstm_prob'],
                            results['final_prob']
                        ]
                    })
                    st.bar_chart(chart_data.set_index('Model'))
                
                # Detailed analysis
                if show_details:
                    st.markdown("---")
                    st.markdown("### 🔬 Detailed Model Breakdown")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Dense NN",
                            results['dense_class'],
                            f"{results['dense_prob']:.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            "CNN",
                            results['cnn_class'],
                            f"{results['cnn_prob']:.4f}"
                        )
                    
                    with col3:
                        st.metric(
                            "BiLSTM",
                            results['bilstm_class'],
                            f"{results['bilstm_prob']:.4f}"
                        )
                    
                    with col4:
                        st.metric(
                            "Meta-Learner",
                            results['final_class'],
                            f"{results['final_prob']:.4f}"
                        )
                
                # Cleaned text preview
                with st.expander("👁️ View Preprocessed Text"):
                    st.text(results['clean_text'][:500] + "..." if len(results['clean_text']) > 500 else results['clean_text'])
            
            else:
                st.error("❌ Models not found")
                st.info("Make sure to run the notebook cell that saves the models first.")
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

else:
    st.info("👉 Enter some news text above to get started!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Fake News Detection System | Powered by Deep Learning
    </div>
""", unsafe_allow_html=True)
