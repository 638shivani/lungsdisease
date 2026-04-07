from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Lung Disease API is running 🚀"}
