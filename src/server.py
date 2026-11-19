from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Praatmaatje")

@app.get("/api/health")
def health():
    """A simple check that the server is running."""
    return {"status": "ok"}

# Serve the web page (web/index.html) at the main address
app.mount("/", StaticFiles(directory="web", html=True), name="web")