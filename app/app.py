import streamlit as st
from get_youtube_comments import get_video_comments, get_video_title
from sentiment_predictor import SentimentPredictor
import pandas as pd
import plotly.express as px

API_KEY = ''

# --- Placeholder Functions ---
st.markdown("""
<style>
.positive-sentiment { /* ... */ }
.negative-sentiment { /* ... */ }
.neutral-sentiment { /* ... */ }
/* ... other styles ... */
</style>
""", unsafe_allow_html=True)

def style_sentiment_cell(val):
    val_lower = str(val).lower() # Ensure val is string before lower()
    if 'positive' in val_lower:
        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    elif 'negative' in val_lower:
        return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
    elif 'neutral' in val_lower:
        return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
    return '' # Return empty string for no styling


# --- END Placeholder Functions ---



# --- START ---
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

                    # The predictor based on our trained model
                    predictor = SentimentPredictor()          

                    if not comments:
                        st.warning("No comments found for this video, or fetching failed. Cannot perform sentiment analysis.")
                        st.write("*(Note: Some videos disable comments or have no comments yet.)*")
                    else:
                        # Process all comments for sentiment
                        st.subheader("Performing Sentiment Analysis...")
                        # To store structured data for the table
                        all_comment_results = [] 

                        # Use st.progress for visual feedback during processing
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        for i, comment in enumerate(comments):
                            status_text.text(f"Analyzing comment {i+1} of {len(comments)}...")
                            # Predict result for this comment
                            result_dict = predictor.predict(comment)

                            if result_dict == None: 
                                continue

                            all_comment_results.append({
                                "Comment Text": comment,
                                "Sentiment for Product": result_dict.get("sentiment_for_product", "N/A"),
                                "Sentiment for Video": result_dict.get("sentiment_for_video", "N/A")
                            })
                            progress_bar.progress((i + 1) / len(comments))

                        # Clear the status text after completion
                        status_text.empty() 
                        # Clear the progress bar
                        progress_bar.empty() 

                        st.success("Sentiment analysis complete for all comments!")

                        # Display the Comments Table with Sentiments
                        st.subheader("Detailed Comment Analysis")
                        with st.expander(f"Click to view all {len(all_comment_results)} comments with sentiment results"):
                            if all_comment_results:
                                df_comments = pd.DataFrame(all_comment_results)

                                styled_df = df_comments.style.applymap(
                                    style_sentiment_cell,
                                    subset=['Sentiment for Product', 'Sentiment for Video']
                                )

                                st.dataframe(styled_df, use_container_width=True, height=500)
                                # st.dataframe(df_comments, use_container_width=True)
                            else:
                                st.info("No comments were analyzed.")

                        # --- Calculate and Display Overall Sentiment Scores ---
                        st.markdown("---") # Visual separator
                        st.subheader("📊 Overall Sentiment Scores (out of 100)")

                        if all_comment_results:
                            # Convert results to DataFrame for easier counting
                            df_overall_sentiments = pd.DataFrame(all_comment_results)

                            # --- Product Sentiment Score Calculation ---
                            product_counts = df_overall_sentiments['Sentiment for Product'].value_counts()
                            product_pos = product_counts.get('positive', 0)
                            product_neu = product_counts.get('neutral', 0)
                            product_neg = product_counts.get('negative', 0)
                            total_product_comments = product_pos + product_neu + product_neg

                            if total_product_comments > 0:
                                # Raw score calculation (positive=1, neutral=0, negative=-1)
                                raw_product_score = (product_pos * 1) + (product_neu * 0.5) + (product_neg * -1.5)

                                # Normalize to 0-100 scale
                                min_raw_score = -total_product_comments
                                max_raw_score = total_product_comments
                                product_overall_score = ((raw_product_score - min_raw_score) / (max_raw_score - min_raw_score)) * 100
                            else:
                                # Default if no comments
                                product_overall_score = 0.0 

                            # --- Video Sentiment Score Calculation ---
                            video_counts = df_overall_sentiments['Sentiment for Video'].value_counts()
                            video_pos = video_counts.get('positive', 0)
                            video_neu = video_counts.get('neutral', 0)
                            video_neg = video_counts.get('negative', 0)
                            total_video_comments = video_pos + video_neu + video_neg

                            if total_video_comments > 0:
                                # Raw score calculation (positive=1, neutral=0, negative=-1)
                                raw_video_score = (video_pos * 1) + (video_neu * 0.5) + (video_neg * -1.5)

                                # Normalize to 0-100 scale
                                min_raw_score_video = -total_video_comments # Could be different if comment sets differ
                                max_raw_score_video = total_video_comments
                                video_overall_score = ((raw_video_score - min_raw_score_video) / (max_raw_score_video - min_raw_score_video)) * 100
                            else:
                                video_overall_score = 0.0 # Default if no comments

                            # --- Display Scores Inline ---
                            col1, col2 = st.columns(2) # Create two columns for inline display

                            with col1:
                                st.metric(label="Product Overall Score", value=f"{product_overall_score:.1f}", delta=None)
                                # You can add a delta if you have a previous score to compare against

                            with col2:
                                st.metric(label="Video Overall Score", value=f"{video_overall_score:.1f}", delta=None)

                            st.markdown("---") # Another separator

                        else:
                            st.info("No comments were analyzed to compute overall scores.")

                        # Percentage of each value
                        st.subheader("📊 Overall Sentiment Summary")

                        product_sentiments = [res["Sentiment for Product"] for res in all_comment_results]
                        video_sentiments = [res["Sentiment for Video"] for res in all_comment_results]

                        if product_sentiments and video_sentiments:
                            st.markdown("#### Product Sentiment Overview")
                            product_counts = pd.Series(product_sentiments).value_counts()
                            total_product = product_counts.sum()
                            for sentiment_type in ['positive', 'neutral', 'negative']:
                                count = product_counts.get(sentiment_type, 0)
                                percentage = (count / total_product) * 100 if total_product > 0 else 0
                                st.metric(label=f"Product: {sentiment_type.capitalize()}", value=f"{percentage:.1f}%")

                            st.markdown("#### Video Sentiment Overview")
                            video_counts = pd.Series(video_sentiments).value_counts()
                            total_video = video_counts.sum()
                            for sentiment_type in ['positive', 'neutral', 'negative']:
                                count = video_counts.get(sentiment_type, 0)
                                percentage = (count / total_video) * 100 if total_video > 0 else 0
                                st.metric(label=f"Video: {sentiment_type.capitalize()}", value=f"{percentage:.1f}%")

                            # Optional: Add charts here as well for overall summary
                            # Example: A pie chart for product sentiment
                            
                            # product_df = pd.DataFrame({'Sentiment': product_counts.index, 'Count': product_counts.values})
                            # fig_product = px.pie(product_df, values='Count', names='Sentiment', title='Product Sentiment Distribution')
                            # st.plotly_chart(fig_product, use_container_width=True)


                        else:
                            st.info("No sentiments to summarize from the analyzed comments.")


                except Exception as e:
                    st.error(f"An unexpected error occurred during analysis: {e}")
                    st.exception(e) # Displays full traceback for debugging
    else:
        st.warning("Please enter a YouTube video URL to proceed.")

