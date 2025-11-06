from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

app = FastAPI(title="Feed Recommendation API (MongoDB-Compatible)")


# ---------------------------
# Models (match your MongoDB schema)
# ---------------------------

class Post(BaseModel):
    id: int
    content: str
    likes: int
    views: int
    created_at: str


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

def days_since(date_str):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return (datetime.now() - date).days + 1


def recommend_posts_for_user(user, posts):
    # Combine all interests into one string for TF-IDF
    interests_text = " ".join(user["interests"]) if user.get("interests") else ""

    corpus = [interests_text] + [p["content"] for p in posts]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)

    user_vec = tfidf[0:1]
    post_vecs = tfidf[1:]
    similarities = cosine_similarity(user_vec, post_vecs).flatten()

    recommendations = []
    for i, post in enumerate(posts):
        similarity = similarities[i]
        likes = post.get("likes", 0)
        views = post.get("views", 0)
        engagement = (likes * 0.7 + views * 0.3) / 100
        recency = 1 / days_since(post["created_at"])
        score = 0.6 * similarity + 0.3 * engagement + 0.1 * recency

        recommendations.append({
            "post_id": post["id"],
            "content": post["content"],
            "score": round(score, 4),
            "similarity": round(similarity, 3),
            "engagement": round(engagement, 3),
            "recency": round(recency, 3)
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
