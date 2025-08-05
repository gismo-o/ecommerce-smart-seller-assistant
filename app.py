import streamlit as st
import pandas as pd
from preprocessing import predict_sentiment
from llm_gemini import generate_gemini_recommendation
from tts_elevenlabs import generate_audio

# App config
st.set_page_config(page_title="Smart Seller Assistant", layout="wide")
st.sidebar.title("🧭 Navigation")

# Load data
try:
    df = pd.read_csv("data/final_cleaned_data_sample.csv")
except FileNotFoundError:
    st.error("❌ Dataset not found. Please make sure 'data/final_cleaned_data.csv' exists.")
    st.stop()

# Sidebar - Product selection
product_options = df["ProductId"].unique()
selected_product = st.sidebar.selectbox("Select a product", product_options)
product_reviews = df[df["ProductId"] == selected_product]

# Sidebar - Page selection
page = st.sidebar.radio("Select view", ["Overview", "Reviews", "Insights", "Audio"])

# Page: Overview
if page == "Overview":
    st.title("📦 AI-Powered Smart Seller Assistant")
    st.markdown("""
        This app analyzes customer reviews of products to generate actionable, strategic suggestions for sellers.
        Select a product from the sidebar to get started.
    """)
    st.metric(label="Total Reviews", value=len(product_reviews))

# Page: Reviews
elif page == "Reviews":
    st.header("📝 Top 10 Customer Reviews")
    if product_reviews.empty:
        st.warning("No reviews available for this product.")
    else:
        for i, review in enumerate(product_reviews["Clean_Text"].head(10), 1):
            st.markdown(f"**{i}.** {review}")

# Page: Insights
elif page == "Insights":
    st.header("🧠 Analyze and Generate Suggestions")
    if product_reviews.empty:
        st.warning("No reviews to analyze.")
        st.stop()

    # Sentiment analysis
    st.info("🔍 Classifying reviews...")
    try:
        sentiments = predict_sentiment(product_reviews)
        product_reviews["Predicted_Sentiment"] = sentiments
    except Exception as e:
        st.error(f"❌ Error during sentiment prediction: {e}")
        st.stop()

    # Sentiment summary
    sentiment_summary_df = product_reviews["Predicted_Sentiment"].value_counts(normalize=True).mul(100).round(2)
    sentiment_summary = "\n".join([f"{sent}: {pct}%" for sent, pct in sentiment_summary_df.items()])

    # Review summary
    review_summary = "\n".join(product_reviews["Clean_Text"].tolist()[:20])

    # Generate recommendations
    st.success("🤖 Generating strategic suggestions with Gemini...")
    try:
        recommendation = generate_gemini_recommendation(
            selected_product,
            review_summary,
            sentiment_summary
        )
        st.subheader("💡 Strategic Suggestions")
        st.write(recommendation)
        st.session_state.recommendation = recommendation  # Store for audio tab
    except Exception as e:
        st.error(f"❌ Failed to get suggestion from Gemini: {e}")
        st.stop()

# Page: Audio
elif page == "Audio":
    st.header("🔊 Audio Summary")
    recommendation = st.session_state.get("recommendation", None)
    if recommendation:
        try:
            audio_path = generate_audio(recommendation)
            st.audio(audio_path, format="audio/mp3")
        except Exception as e:
            st.error(f"❌ Failed to generate audio: {e}")
    else:
        st.info("First run the 'Insights' tab to generate a recommendation.")

