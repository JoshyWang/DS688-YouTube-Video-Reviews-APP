pip install google-api-python-client nltk wordcloud matplotlib streamlit pandas

import streamlit as st
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from datetime import datetime

# Initialize VADER and YouTube API
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

analyzer = SentimentIntensityAnalyzer()
API_KEY = 'AIzaSyDNd-_-13SkXq8Uzo1zyErK89BVG_GCQE4'  # Replace with your YouTube API key
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Helper functions for sentiment analysis, lemmatization, etc.
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to get comments
def get_comments(video_id, min_comments=100):
    comments = []
    next_page_token = None
    while len(comments) < min_comments:
        try:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                textFormat='plainText',
                pageToken=next_page_token
            ).execute()
            comments.extend([item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']])
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            st.write(f"Error retrieving comments for video {video_id}: {e}")
            break
    return comments

# Function to process comments for lemmatized word cloud
def process_comments(comments):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = []
    for comment in comments:
        comment = re.sub('[^a-zA-Z]', ' ', comment).lower()
        words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in nltk.pos_tag(word_tokenize(comment)) if word not in stop_words]
        tokens.extend(words)
    return ' '.join(tokens)

# Function to analyze sentiment and get compound score
def analyze_sentiment(comments):
    scores = {'compound': 0}
    total_comments = len(comments)
    for comment in comments:
        score = analyzer.polarity_scores(comment)
        scores['compound'] += score['compound']
    scores['compound'] /= total_comments if total_comments > 0 else 1
    return scores['compound']

# Streamlit app starts here
st.title("YouTube Video Analysis App")

st.sidebar.header("Enter Search Parameters")
start_date = st.sidebar.date_input("Start Date", datetime.now().replace(day=1))
end_date = st.sidebar.date_input("End Date", datetime.now())
keyword = st.sidebar.text_input("Keyword")

if st.sidebar.button("Analyze"):
    start_date_iso = start_date.isoformat() + "T00:00:00Z"
    end_date_iso = end_date.isoformat() + "T23:59:59Z"
    response = youtube.search().list(
        part="snippet",
        q=keyword,
        type="video",
        order="viewCount",
        publishedAfter=start_date_iso,
        publishedBefore=end_date_iso,
        maxResults=10
    ).execute()

    # Analyze each video
    best_video = None
    best_score = -1
    for item in response['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        comments = get_comments(video_id)
        compound_score = analyze_sentiment(comments)
        
        if compound_score > best_score:
            best_score = compound_score
            best_video = {
                'video_id': video_id,
                'title': title,
                'compound_score': compound_score,
                'comments': comments
            }

    # Display best video details and word cloud
    if best_video:
        st.write("### Video with Highest Compound Sentiment Score")
        st.write(f"**Title:** {best_video['title']}")
        st.write(f"**Compound Sentiment Score:** {best_video['compound_score']}")
        st.write(f"https://www.youtube.com/watch?v={best_video['video_id']}")
        
        # Generate word cloud
        preprocessed_comments = process_comments(best_video['comments'])
        generate_word_cloud(preprocessed_comments)
    else:
        st.write("No videos found for the given parameters.")
