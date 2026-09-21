# ------------------------------------------------------------------
# APPLICATION / PREDICTION SCRIPT: app.py
#
# Run with:
#     python app.py
#
# Loads the saved model and vectorizer once at startup, then exposes a
# /predict endpoint that classifies any email text sent to it.
# The trained artifacts must already exist, so run train_model.py first.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# MODULE IMPORTS
# ------------------------------------------------------------------
import os                                            # Builds OS-independent file paths
import joblib                                        # Loads the saved .pkl artifacts back into memory
import uvicorn                                       # ASGI server that runs the FastAPI app
from fastapi import FastAPI, HTTPException           # Web framework and HTTP error helper
from fastapi.middleware.cors import CORSMiddleware   # Lets the HTML frontend call this API from the browser
from pydantic import BaseModel, Field                # Declares and validates the shape of incoming JSON


# ------------------------------------------------------------------
# STEP 1: INITIALIZE THE FASTAPI APPLICATION
# ------------------------------------------------------------------
# The title and description show up in the automatic interactive docs at
# http://127.0.0.1:8000/docs once the server is running.
app = FastAPI(
    title="Email Spam Detection API",
    description="Supervised machine learning backend (TF-IDF + Multinomial Naive Bayes)",
    version="1.0.0"
)


# ------------------------------------------------------------------
# STEP 2: CONFIGURE CORS
# ------------------------------------------------------------------
# Browsers block a page from calling a server on a different origin unless that server
# explicitly allows it. The frontend is opened as a local file or from a different port,
# so without this middleware every request from index.html would be rejected.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Any origin may call this API (fine for a local learning project)
    allow_credentials=True,   # Permit cookies and credential headers
    allow_methods=["*"],      # Permit GET, POST, OPTIONS, etc.
    allow_headers=["*"],      # Permit all custom request headers
)


# ------------------------------------------------------------------
# STEP 3: LOAD THE TRAINED ARTIFACTS ONCE AT STARTUP
# ------------------------------------------------------------------
# Loading happens a single time when the server boots, NOT on every request.
# Retraining or reloading per request would be slow and completely unnecessary.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)            # The trained Naive Bayes classifier
    vectorizer = joblib.load(VECTORIZER_PATH)  # The fitted TF-IDF vectorizer
    print("Model and vectorizer loaded successfully.")
except Exception as e:
    # Fail loudly and early with a useful message rather than crashing on the first request
    raise RuntimeError(
        f"Could not load model files: {e}\n"
        "Run 'python train_model.py' first to create spam_model.pkl and vectorizer.pkl."
    )


# ------------------------------------------------------------------
# STEP 4: DEFINE THE INPUT VALIDATION SCHEMA
# ------------------------------------------------------------------
# Pydantic checks every incoming request body automatically. If a client sends the wrong
# type or an empty string, FastAPI returns a clear 422 error before our code ever runs.
class EmailInput(BaseModel):
    email_text: str = Field(
        ...,                 # The ellipsis marks this field as required
        min_length=1,
        description="The raw email or message text to classify"
    )


# ------------------------------------------------------------------
# STEP 5: ROUTE - HEALTH CHECK
# ------------------------------------------------------------------
# A simple GET endpoint to confirm the server is alive, useful for debugging.
@app.get("/")
def home():
    return {
        "status": "active",
        "message": "Email Spam Detection API is running",
        "docs": "/docs"
    }


# ------------------------------------------------------------------
# STEP 6: ROUTE - PREDICTION ENDPOINT
# ------------------------------------------------------------------
# Accepts JSON like {"email_text": "Win a free prize now!"} and returns the classification.
@app.post("/predict")
def predict_spam(data: EmailInput):
    try:
        # Reject whitespace-only input that passed the length check
        text = data.email_text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Email text cannot be empty.")

        # STEP 6a: Apply the SAME preprocessing used during training.
        # .transform() (never .fit_transform()) reuses the saved vocabulary so the new
        # email lands in exactly the same feature columns the model was trained on.
        features = vectorizer.transform([text])

        # STEP 6b: Run the trained classifier.
        # [0] pulls the single result out of the returned array, since we sent one email.
        prediction = int(model.predict(features)[0])

        # STEP 6c: Ask for class probabilities as well, so the UI can show confidence.
        # predict_proba returns [probability_of_ham, probability_of_spam].
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])

        # STEP 6d: Translate the numeric label back into a human-readable answer.
        label = "Spam" if prediction == 1 else "Not Spam"

        # Return a structured JSON response for the frontend to display
        return {
            "prediction": prediction,                          # 1 = spam, 0 = not spam
            "label": label,                                    # Human-readable class name
            "confidence": round(confidence * 100, 2),          # Confidence in the chosen class, as a percentage
            "spam_probability": round(float(probabilities[1]) * 100, 2),
            "email_text": text
        }

    except HTTPException:
        # Re-raise our own deliberate errors untouched
        raise
    except Exception as e:
        # Any unexpected failure becomes a clean 500 response instead of a stack trace
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ------------------------------------------------------------------
# STEP 7: START THE SERVER
# ------------------------------------------------------------------
# This block runs only when the file is executed directly (python app.py),
# not when it is imported by another module.
if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:8000")
    print("Interactive API docs: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)
