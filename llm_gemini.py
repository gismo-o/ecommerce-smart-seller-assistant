import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def generate_gemini_recommendation(product_title, review_summary, sentiment_summary):
    prompt = f"""
    Product: {product_title}

    Below is a sentiment analysis summary and key excerpts from user reviews.

    Sentiment Distribution:
    {sentiment_summary}

    Sample Reviews:
    {review_summary}

    Based on this data, generate **short, clear, and actionable suggestions** for the seller.

    Focus on:
    - How to address common negative feedback
    - How to leverage strengths highlighted in positive reviews
    - Improvements that can increase customer satisfaction and trust

    Tone: professional, concise, and strategic.
    Format: bullet points.
    Language: English.
    """

    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text

