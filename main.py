from dotenv import load_dotenv
from fastapi import FastAPI

from middlewares.auth import auth_middleware
from routes.submission import router as submission_router
from routes.mastery import router as mastery_router
from routes.recommendation import router as recommendation_router
from routes.seeding import router as seeding_router

load_dotenv()

app = FastAPI()

app.middleware("http")(auth_middleware)

app.include_router(submission_router)
app.include_router(mastery_router)
app.include_router(recommendation_router)
app.include_router(seeding_router)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Recommendation Service"
    }