import googleapiclient.discovery
import googleapiclient.errors
import re
import csv
import os

def extract_video_id(youtube_url):
    """
    Extracts the video ID from a YouTube URL.
    """
    if "youtube.com/watch?v=" in youtube_url:
        match = re.search(r"v=([a-zA-Z0-9_-]{11})", youtube_url)
    elif "youtu.be/" in youtube_url:
        match = re.search(r"youtu.be/([a-zA-Z0-9_-]{11})", youtube_url)
    else:
        return None

    if match:
        return match.group(1)
    return None

def get_video_comments(video_url, api_key, max_results=100):
    """
    Fetches comments for a given YouTube video URL using the YouTube Data API.

    Args:
        video_url (str): The URL of the YouTube video.
        api_key (str): Your Google API key.
        max_results (int): The maximum number of top-level comments to retrieve per page (max 100).

    Returns:
        comments: A list of comments.
        
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Invalid YouTube URL.")
        return []

    comments = []
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=max_results,
            order="relevance" # or "time" for chronological order
        )
        response = request.execute()

        while response:
            for item in response["items"]:
                comment_data = item["snippet"]["topLevelComment"]["snippet"]
                comments.append(comment_data["textOriginal"])

            if "nextPageToken" in response:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=max_results,
                    pageToken=response["nextPageToken"],
                    order="relevance"
                )
                response = request.execute()
            else:
                break

    except googleapiclient.errors.HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        print("Please check your API key, video ID, and quota limits.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return comments

def get_video_title(video_url, api_key):
    """
    Fetches the title of a YouTube video given its URL using the YouTube Data API.

    Args:
        video_url (str): The URL of the YouTube video.
        api_key (str): Your Google API key.

    Returns:
        video_title: A string of the video title.
    """
    video_id = extract_video_id(video_url)
    if not video_id:
        print("Invalid YouTube URL provided.")
        return None

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    try:
        request = youtube.videos().list(
            part="snippet", # We only need the snippet part for the title
            id=video_id
        )
        response = request.execute()

        if response and response["items"]:
            video_title = response["items"][0]["snippet"]["title"]
            return video_title
        else:
            print(f"Video with ID '{video_id}' not found or no items in response.")
            return None

    except googleapiclient.errors.HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")
        print("Please check your API key, video ID, and quota limits.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
    



if __name__ == "__main__":
    # Replace with your YouTube video URL and API key
    YOUTUBE_VIDEO_URL = "https://youtu.be/1k4EQaBiOZc?si=wVOOcKxk_dZJhAiu" # Example: Rick Astley - Never Gonna Give You Up
    YOUR_API_KEY = "AIzaSyAovCSJvOyVdtxFPEv7Kf30CZMfEaEWElc"  # <-- **IMPORTANT: Replace with your actual API key**

    video_id = extract_video_id(YOUTUBE_VIDEO_URL)


    if YOUR_API_KEY == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual YouTube Data API key.")
    else:
        print(f"Fetching comments for: {YOUTUBE_VIDEO_URL}")
        video_comments = get_video_comments(YOUTUBE_VIDEO_URL, YOUR_API_KEY, max_results=100)
        video_title = get_video_title(YOUTUBE_VIDEO_URL, YOUR_API_KEY)
        
        video_title = re.sub('[^a-zA-Z0-9]', ' ', video_title.strip())
        video_title = re.sub('\s+', '_', video_title.strip())
        video_title = video_title.lower()

        if video_comments:
            print(f"\nFound {len(video_comments)} comments:")
            for i, comment in enumerate(video_comments[:10]): # Print first 10 comments as a preview
                print(f"--- Comment {i+1} ---")
                print(f"Text: {comment[:150]}...") # Truncate long comments for display

            os.makedirs("data/comments_unlabelled", exist_ok=True)
            
            with open("data/comments_unlabelled/comments_" + video_title + ".csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # writer.writeheader()
                writer.writerow(['text', 'sentiment_for_product', 'sentiment_for_video'])
                for comment in video_comments:
                    if comment.strip() == '': 
                        continue
                    comment.replace('\n', '. ')
                    writer.writerow([comment, '', ''])

        else:
            print("No comments found or an error occurred.")