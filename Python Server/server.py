from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return "This is python API for dna mutation prediction"

@app.get("/about")
def about():
    return "This is about page"