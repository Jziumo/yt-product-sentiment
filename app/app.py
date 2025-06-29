import streamlit as st
from get_youtube_comments import get_video_comments, get_video_title
# app/app.py

API_KEY = 'AIzaSyAovCSJvOyVdtxFPEv7Kf30CZMfEaEWElc'

# Assuming these modules exist in your src directory and have the respective functions
# from src.data.youtube_collector import get_youtube_comments
# from src.data.preprocessor import preprocess_text
# from src.model.sentiment_model import analyze_sentiment

# --- Placeholder Functions (REMOVE IN YOUR ACTUAL PROJECT) ---
# These are just to make the app runnable for demonstration purposes
# You MUST replace these with your actual functions from src/
def get_youtube_comments(url):
    """Simulates fetching comments."""
    st.info(f"Simulating fetching comments for: {url}")
    # In a real app, you'd use google-api-python-client here
    if "error" in url: # Simulate an error
        raise ValueError("Simulated error during comment fetching.")
    return [
        "This product is amazing! Highly recommend.",
        "It's okay, nothing special. A bit overpriced.",
        "Absolutely terrible, wasted my money.",
        "Good value for money, very useful.",
        "I'm neutral about it. It does the job.",
        "Worst purchase ever. Stay away!",
        "Fantastic quality and fast delivery.",
        "Not bad, but could be better.",
        "Love it! Changed my life.",
        "Meh. Disappointed.",
    ]

def preprocess_text(text):
    """Simulates text preprocessing."""
    st.info(f"Simulating preprocessing: '{text[:30]}...'")
    # In a real app, you'd clean, tokenize, remove stopwords, etc.
    return text.lower()

def analyze_sentiment(text):
    """Simulates sentiment analysis."""
    st.info(f"Simulating sentiment for: '{text[:30]}...'")
    # In a real app, you'd use your pre-trained model (e.g., Hugging Face, NLTK, TextBlob)
    if "amazing" in text or "recommend" in text or "fantastic" in text or "love" in text:
        return {'label': 'positive', 'score': 0.9}
    elif "terrible" in text or "wasted" in text or "worst" in text or "disappointed" in text:
        return {'label': 'negative', 'score': 0.8}
    else:
        return {'label': 'neutral', 'score': 0.6}
# --- END Placeholder Functions ---


# --- Streamlit App Layout ---

# Set wide layout for better display of results (optional)
st.set_page_config(layout="wide", page_title="YouTube Review Sentiment Analyzer")

# Custom Title (can be improved with emojis or markdown)
st.title("🎬 YouTube Product Review Sentiment Analyzer")

st.markdown(
    """
    Enter a YouTube video URL (preferably a product review) below to analyze the sentiment
    of its comments.
    """
)

# Input for YouTube URL
youtube_url = st.text_input(
    "Paste YouTube Video URL here:",
    placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    key="url_input" # Unique key for this widget
)

# Button to trigger analysis
if st.button("Analyze Video Comments", key="analyze_button"):
    # comments = get_video_comments(youtube_url)
    # title = get_video_title(youtube_url)

    if youtube_url:
        # Basic URL validation (you might want more robust regex here)
        if "youtube.com/watch" not in youtube_url and "youtu.be/" not in youtube_url:
            st.error("Please enter a valid YouTube video URL.")
        else:
            with st.spinner("Fetching video details and comments..."):
                try:
                    # Fetch comments and title
                    comments = get_video_comments(youtube_url, api_key=API_KEY) 
                    title = get_video_title(youtube_url, api_key=API_KEY)

                    st.success("Video details and comments fetched!")

                    # --- Display Video Title ---
                    if title:
                        st.subheader(f"Video Title: {title}")
                        # A link to the video page. 
                        st.markdown(f"[Watch Video]({youtube_url})")
                    else:
                        st.warning("Could not retrieve video title.")

                    # --- Display Comments ---
                    st.subheader("Raw Comments:")
                    if comments:
                        # Display comments in an expandable section for neatness
                        with st.expander(f"View {len(comments)} Comments"):
                            for i, comment in enumerate(comments[:10]): # Show first 10
                                st.write(f"**Comment {i+1}:** {comment}")
                            if len(comments) > 10:
                                st.info(f"... and {len(comments) - 10} more comments.")
                    else:
                        st.info("No comments found for this video.")

                    # --- Continue with Sentiment Analysis (from previous code) ---
                    # You would put the rest of your sentiment analysis logic here,
                    # which uses 'comments' (the actual comments fetched).

                    # if comments: # Only proceed with sentiment analysis if comments were fetched
                    #     st.subheader("📊 Sentiment Analysis Results")
                    #     sentiment_results = []
                    #     for i, comment in enumerate(comments):
                    #         # Display progress in the spinner area
                    #         st.spinner(f"Processing comment {i+1}/{len(comments)} for sentiment...")
                    #         processed_comment = preprocess_text(comment)
                    #         sentiment = analyze_sentiment(processed_comment)
                    #         sentiment_results.append({
                    #             'original_comment': comment,
                    #             'processed_comment': processed_comment,
                    #             'sentiment_label': sentiment['label'],
                    #             'sentiment_score': sentiment['score']
                    #         })

                    #     # Calculate counts
                    #     positive_count = sum(1 for res in sentiment_results if res['sentiment_label'] == 'positive')
                    #     negative_count = sum(1 for res in sentiment_results if res['sentiment_label'] == 'negative')
                    #     neutral_count = sum(1 for res in sentiment_results if res['sentiment_label'] == 'neutral')
                    #     total_analyzed = len(sentiment_results)

                    #     if total_analyzed > 0:
                    #         st.write(f"**Total Comments Analyzed:** {total_analyzed}")
                    #         col1, col2, col3 = st.columns(3) # Use columns for metrics
                    #         with col1:
                    #             st.metric(label="👍 Positive Reviews", value=f"{positive_count} ({(positive_count/total_analyzed)*100:.1f}%)")
                    #         with col2:
                    #             st.metric(label="👎 Negative Reviews", value=f"{negative_count} ({(negative_count/total_analyzed)*100:.1f}%)")
                    #         with col3:
                    #             st.metric(label="😐 Neutral Reviews", value=f"{neutral_count} ({(neutral_count/total_analyzed)*100:.1f}%)")

                    #         # Optional: Display a chart
                    #         import pandas as pd
                    #         import plotly.express as px

                    #         sentiment_data = pd.DataFrame({
                    #             'Sentiment': ['Positive', 'Negative', 'Neutral'],
                    #             'Count': [positive_count, negative_count, neutral_count]
                    #         })
                    #         fig = px.pie(sentiment_data, values='Count', names='Sentiment', title='Distribution of Sentiments',
                    #                      color_discrete_map={'Positive':'#28a745', 'Negative':'#dc3545', 'Neutral':'#ffc107'})
                    #         st.plotly_chart(fig, use_container_width=True)

                    #         st.subheader("💬 Individual Comment Sentiment")
                    #         df_results = pd.DataFrame(sentiment_results)
                    #         st.dataframe(df_results[['original_comment', 'sentiment_label', 'sentiment_score']], use_container_width=True)
                    #     else:
                    #         st.info("No sentiments to summarize from the fetched comments.")
                    # else:
                    #     st.info("Cannot perform sentiment analysis as no comments were fetched.")

                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {e}")
                    st.exception(e) # Displays full traceback for debugging
    else:
        st.warning("Please enter a YouTube video URL to proceed.")
    