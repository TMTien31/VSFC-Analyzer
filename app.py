# vsfc_app/app.py
import streamlit as st
import pandas as pd
from utils.text_utils import split_sentences
from utils.predictor import SentimentPredictor, TopicPredictor
import plotly.express as px

# Load models
sentiment_model = SentimentPredictor("tmt3103/VSFC-sentiment-classify-phoBERT")
topic_model = TopicPredictor("tmt3103/VSFC-topic-classify-phoBERT")

st.title("VSFC Evaluation Analyzer")

text_input = st.text_area("Enter feedback (separated by dots '.')")

if st.button("Analyze"):
    sentences = split_sentences(text_input)
    sentiments = sentiment_model.predict(sentences)
    topics = topic_model.predict(sentences)

    df = pd.DataFrame({
        "Sentence": sentences,
        "Topic": topics,
        "Sentiment": sentiments
    })

    st.subheader("Classification Result")
    st.dataframe(df)

    st.subheader("Sentiment distribution")
    fig1 = px.histogram(df, x="Sentiment", color="Sentiment", barmode="group")
    st.plotly_chart(fig1)

    st.subheader("Topic distribution")
    fig2 = px.histogram(df, x="Topic", color="Topic", barmode="group")
    st.plotly_chart(fig2)

    st.subheader("Heatmap")
    crosstab = pd.crosstab(df["Topic"], df["Sentiment"])
    st.dataframe(crosstab)
    fig3 = px.imshow(crosstab, text_auto=True, aspect="auto")
    st.plotly_chart(fig3)

    st.subheader("Typical sentences by topic + sentiment")
    for topic in df["Topic"].unique():
        for sentiment in df["Sentiment"].unique():
            filtered = df[(df["Topic"] == topic) & (df["Sentiment"] == sentiment)]
            if not filtered.empty:
                st.markdown(f"**{topic} - {sentiment}**")
                st.write(filtered["Sentence"].iloc[0])
