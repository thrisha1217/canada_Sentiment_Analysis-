import streamlit as st
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Set app title
st.title("Canada Wikipedia Sentiment Analysis 🇨🇦")

st.markdown("""
This app analyzes the sentiment of user input based on content scraped from the **Canada Wikipedia page**.
""")

# Scrape Wikipedia content
@st.cache_data
def get_wikipedia_content(url="https://en.wikipedia.org/wiki/Canada"):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = " ".join([para.get_text() for para in paragraphs])
    return text

wiki_text = get_wikipedia_content()

# User input
user_input = st.text_area("Enter your sentence to analyze:")

if user_input:
    # ✅ Safe WordCloud block
    try:
        if len(user_input.strip()) > 2:
            wordcloud = WordCloud(
                width=800, height=400, background_color='white', stopwords=STOPWORDS
            ).generate(user_input)
            st.image(wordcloud.to_array(), caption="🌥️ Word Cloud", use_container_width=True)
        else:
            st.warning("Please enter a longer sentence for WordCloud to generate.")
    except Exception as e:
        st.error(f"⚠️ Could not generate WordCloud: {e}")

    # TextBlob Analysis
    blob = TextBlob(user_input)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    st.markdown("### 📊 TextBlob Sentiment Analysis")
    st.write(f"**Polarity**: `{polarity:.2f}`")
    st.write(f"**Subjectivity**: `{subjectivity:.2f}`")

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    sns.barplot(x=['Polarity', 'Subjectivity'], y=[polarity, subjectivity], palette='coolwarm')
    ax2.set_ylim(-1, 1)
    ax2.set_title('TextBlob Sentiment Metrics')
    st.pyplot(fig2)

    # Optional: load trained model and predict (if available)
    if st.button("Run TF-IDF + Predict"):
        try:
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('sentiment_model.pkl', 'rb') as f:
                model = pickle.load(f)

            user_vector = vectorizer.transform([user_input])
            pred = model.predict(user_vector)[0]
            proba = model.predict_proba(user_vector)[0]

            st.success(f"🔮 Predicted Sentiment: `{pred}`")
            fig, ax = plt.subplots()
            ax.bar(['Negative', 'Positive'], proba, color=['red', 'green'])
            ax.set_title("Model Confidence")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        except Exception as e:
            st.warning("🚧 Model not found. Please ensure 'tfidf_vectorizer.pkl' and 'sentiment_model.pkl' exist.")

# About
st.subheader("📌 About")
st.markdown("""
- This app scrapes and analyzes the **Wikipedia page of Canada**
- It uses `TextBlob` for sentiment scoring
- Optional machine learning integration with saved models (TF-IDF + Classifier)
- You can retrain using your pipeline (SMOTE, TF-IDF, and classifiers)

**Built with ❤️ using Streamlit, NLTK, TextBlob, Sklearn**
""")
