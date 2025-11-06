from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

app = FastAPI(title="Feed Recommendation API (MongoDB-Compatible)")


# ---------------------------
# MongoDB-Compatible Models
# ---------------------------

class Post(BaseModel):
    id: str                              # MongoDB ObjectId as string
    user: Optional[str] = None            # Post author ObjectId
    title: str
    description: str
    category: str
    mediaType: Optional[str] = None
    mediaUrl: Optional[str] = None
    likes: Optional[List[str]] = []       # List of user ObjectIds
    dislikes: Optional[List[str]] = []    # List of user ObjectIds
    created_at: str                       # ISO format (e.g. "2025-11-01T12:00:00")


class User(BaseModel):
    name: str
    email: str
    branch: str
    interests: List[str]
    year: str


class RequestData(BaseModel):
    user: User
    posts: List[Post]


# ---------------------------
# Helper functions
# ---------------------------

def days_since(date_str: str) -> int:
    """
    Calculate days since the post was created.
    Accepts both 'YYYY-MM-DD' and full ISO date strings.
    """
    try:
        date = datetime.fromisoformat(date_str.replace("Z", ""))
    except ValueError:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    return (datetime.now() - date).days + 1


def recommend_posts_for_user(user, posts):
    """
    Recommend posts based on:
    - Content similarity (TF-IDF on title, description, category)
    - Engagement (likes/dislikes ratio)
    - Recency (newer posts are ranked higher)
    """
    interests_text = " ".join(user.get("interests", []))

    # Combine content fields
    corpus = [interests_text] + [
        f"{p['title']} {p['description']} {p.get('category', '')}" for p in posts
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)

    user_vec = tfidf[0:1]
    post_vecs = tfidf[1:]
    similarities = cosine_similarity(user_vec, post_vecs).flatten()

    recommendations = []
    for i, post in enumerate(posts):
        similarity = similarities[i]
        likes = len(post.get("likes", []))
        dislikes = len(post.get("dislikes", []))
        total_engagement = likes + dislikes

        # Engagement score (weighted likes)
        engagement = (likes - 0.5 * dislikes) / (total_engagement + 1)

        # Recency (newer posts get more weight)
        recency = 1 / days_since(post["created_at"])

        # Weighted scoring
        score = 0.6 * similarity + 0.3 * engagement + 0.1 * recency

        recommendations.append({
            "post_id": post["id"],
            "title": post["title"],
            "description": post["description"],
            "category": post["category"],
            "mediaType": post.get("mediaType"),
            "mediaUrl": post.get("mediaUrl"),
            "score": round(score, 4),
            "similarity": round(similarity, 3),
            "engagement": round(engagement, 3),
            "recency": round(recency, 3),
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations


# ---------------------------
# API Endpoint
# ---------------------------

@app.post("/recommend")
def get_recommendations(data: RequestData):
    user = data.user.dict()
    posts = [p.dict() for p in data.posts]
    recommendations = recommend_posts_for_user(user, posts)
    return {"recommendations": recommendations}
