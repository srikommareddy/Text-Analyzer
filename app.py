import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

# Download NLTK resources (only first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

st.set_page_config(page_title="Text Processing App", layout="wide")

st.title("📑 Text Processing ML App")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    st.subheader("📖 Original Text Preview")
    st.text_area("Text Content", text[:2000] + "..." if len(text) > 2000 else text, height=200)

    # Tokenization
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [w for w in words if w.isalpha() and w not in stop_words]

    # 1. Word count & Vocabulary
    word_count = len(filtered_words)
    vocabulary = len(set(filtered_words))

    st.subheader("📊 Basic Statistics")
    st.write(pd.DataFrame({
        "Metric": ["Word Count", "Vocabulary Size"],
        "Value": [word_count, vocabulary]
    }))

    # 2. Grammar Statistics
    tagged = pos_tag(filtered_words)
    pos_counts = {"Nouns": 0, "Adjectives": 0, "Verbs": 0, "Adverbs": 0}
    for word, tag in tagged:
        if tag.startswith("NN"):
            pos_counts["Nouns"] += 1
        elif tag.startswith("JJ"):
            pos_counts["Adjectives"] += 1
        elif tag.startswith("VB"):
            pos_counts["Verbs"] += 1
        elif tag.startswith("RB"):
            pos_counts["Adverbs"] += 1

    pos_df = pd.DataFrame(list(pos_counts.items()), columns=["POS", "Count"])
    st.subheader("📌 Grammar Statistics")
    st.table(pos_df)

    # Bar chart
    bar_chart = alt.Chart(pos_df).mark_bar().encode(
        x='POS',
        y='Count',
        color='POS'
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # 3. Word Cloud
    st.subheader("☁️ Word Cloud")
    wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(filtered_words))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    # 4. Sentiment Analysis
    st.subheader("😊 Sentiment Analysis")
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # -1 to 1
    sentiment_label = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"
    st.write(f"**Sentiment Score:** {sentiment_score:.3f} → {sentiment_label}")
