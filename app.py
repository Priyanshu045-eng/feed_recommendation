from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import motor.motor_asyncio
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="Feed Recommendation API (MongoDB-Compatible)")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["auth-db"]
users_collection = db.users
posts_collection = db.posts

@app.on_event("startup")
async def startup_db_client():
    try:
        await client.admin.command('ping')
        print("✅ MongoDB connected successfully!")
    except Exception as e:
        print("❌ MongoDB connection failed:", e)

class RecommendedPost(BaseModel):
    post_id: str
    title: str
    description: str
    category: str
    mediaType: Optional[str]
    mediaUrl: Optional[str]
    score: float
    similarity: float
    engagement: float
    recency: float

class RecommendPostsResponse(BaseModel):
    recommendations: Optional[List[RecommendedPost]] = None
    error: Optional[str] = None

class UserRequest(BaseModel):
    user_id: str
    top_n: int = 10 



def days_since(date_input) -> int:
    if isinstance(date_input, str):
        try:

            date = datetime.fromisoformat(date_input.replace("Z", ""))
        except ValueError:
    
            date = datetime.strptime(date_input, "%Y-%m-%d")
    elif isinstance(date_input, datetime):
        date = date_input
    else:
        date = datetime.now()
    return max((datetime.now() - date).days, 1)


def calculate_post_score(user_interests_text, posts):
    corpus = [user_interests_text] + [
        f"{p['title']} {p['description']} {p.get('category','')}" for p in posts
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
        engagement = (likes - 0.5 * dislikes) / (total_engagement + 1)
        recency = 1 / days_since(post["createdAt"])
        score = 0.6 * similarity + 0.3 * engagement + 0.1 * recency

        recommendations.append({
            "post_id": str(post["_id"]),
            "title": post["title"],
            "description": post["description"],
            "category": post["category"],
            "mediaType": post.get("mediaType"),
            "mediaUrl": post.get("mediaUrl"),
            "score": round(score, 4),
            "similarity": round(similarity, 3),
            "engagement": round(engagement, 3),
            "recency": round(recency, 3)
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations

@app.post("/recommend_posts/", response_model=RecommendPostsResponse)
async def recommend_posts(request: UserRequest):
    try:
        from bson import ObjectId
        user = await users_collection.find_one({"_id": ObjectId(request.user_id)})
        if not user:
            return {"error": f"User with id {request.user_id} not found."}

        user_interests_text = " ".join(user.get("interests", []))
        posts = await posts_collection.find({}).to_list(length=1000)
        if not posts:
            return {"error": "No posts found in database."}

        recommended_posts = calculate_post_score(user_interests_text, posts)
        return {"recommendations": recommended_posts}
    except Exception as e:
        return {"error": str(e)}
@app.get("/")
def home():
    return {"message": "Backend running successfully on Render 🚀"}


