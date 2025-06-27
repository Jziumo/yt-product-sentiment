import googleapiclient.discovery
import googleapiclient.errors
import re
import csv
import os
import random as rd
import sys

# Replace with your own key or Jin's Key
# For some reasons I cannot directly put it here.
API_KEY = ""

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
    
def save_youtube_comments(video_url, sample_num = 100, api_key=""): 
    """
    Save youtube video comments to a CSV file.
    """
    if api_key == "" and API_KEY != "": 
        api_key = API_KEY

    if api_key == "":
        print("Please replace 'API_KEY' with your actual YouTube Data API key.")
        return
    

    print(f"Fetching comments for: {video_url}")
    video_comments = get_video_comments(video_url, api_key, max_results=100)
    video_title = get_video_title(video_url, api_key)

    print(f'Video title: {video_title}')
    
    video_title = re.sub('[^a-zA-Z0-9]', ' ', video_title.strip())
    video_title = re.sub('\s+', '_', video_title.strip())
    video_title = video_title.lower()


    if video_comments:
        print(f"\nFound {len(video_comments)} comments. ")

        expected_comments_num = min(len(video_comments), sample_num)

        if expected_comments_num < len(video_comments):
            video_comments = rd.sample(video_comments, expected_comments_num)

        print(f'Randomly choose {expected_comments_num} comments as samples.')

        os.makedirs("data/comments_unlabelled", exist_ok=True)
        
        with open("data/comments_unlabelled/comments_" + video_title + ".csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter='\t')
            # writer.writeheader()
            writer.writerow(['text', 'sentiment_for_product', 'sentiment_for_video'])
            for comment in video_comments:
                if comment.strip() == '': 
                    continue
                writer.writerow([comment, '', ''])

    else:
        print("No comments found or an error occurred.")


if __name__ == "__main__":
    # Example for calling the function
    # save_youtube_comments("https://youtu.be/1k4EQaBiOZc?si=wVOOcKxk_dZJhAiu")

    if not (len(sys.argv) == 2 or len(sys.argv) == 3):
        print("Usage: python get_youtube_comments.py <video_title> <sample_num>")
        sys.exit(1)

    video_title = sys.argv[1]
    sample_num = 100 # default

    if len(sys.argv) == 3: 
        try:
            sample_num = int(sys.argv[2])
        except ValueError:
            print("Error: sample_num must be an integer.")
            sys.exit(1)

    save_youtube_comments(video_title, sample_num)