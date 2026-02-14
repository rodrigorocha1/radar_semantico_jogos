from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
load_dotenv()
# 🔑 Configurações
API_KEY = os.environ['CHAVE_API_YOUTUBE']
VIDEO_ID = "RB6IBTlgVUU"
MAX_RESULTS = 100  # máximo por requisição

# Cria o client da API
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=MAX_RESULTS,
            pageToken=next_page_token,
            textFormat="plainText"  # opcional: plainText ou html
        )
        response = request.execute()
        print(response)

        for item in response["items"]:
            comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": comment_snippet["authorDisplayName"],
                "text": comment_snippet["textDisplay"],
                "likes": comment_snippet["likeCount"],
                "published_at": comment_snippet["publishedAt"]
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# 🔹 Testando
all_comments = get_comments(VIDEO_ID)
print(f"Total de comentários: {len(all_comments)}")
for c in all_comments[:10]:  # mostra os 10 primeiros
    print(f"{c['author']}: {c['text']} ({c['likes']} likes)")
