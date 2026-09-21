# ------------------------------------------------------------------
# APPLICATION / BACKEND SCRIPT: app.py
#
# Run with:
#     python app.py
#
# Loads the saved K-Means model, scaler, and segment-name mapping once
# at startup, then serves a small web form. When a listener's details
# are submitted, it scales the input with the SAME scaler used during
# training, predicts a cluster, and displays the listener segment.
# The trained artifacts must already exist, so run train_model.py first.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# MODULE IMPORTS
# ------------------------------------------------------------------
import os                                    # Builds OS-independent file paths
import joblib                                # Loads the saved .pkl artifacts back into memory
from flask import Flask, render_template, request  # Web framework: app, HTML templates, form data


# ------------------------------------------------------------------
# STEP 1: INITIALIZE THE FLASK APPLICATION
# ------------------------------------------------------------------
app = Flask(__name__)


# ------------------------------------------------------------------
# STEP 2: LOAD THE TRAINED ARTIFACTS ONCE AT STARTUP
# ------------------------------------------------------------------
# Loading happens a single time when the server boots, NOT on every request.
# Retraining or reloading per request would be slow and completely unnecessary.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
SEGMENT_NAMES_PATH = os.path.join(BASE_DIR, "segment_names.pkl")

# The feature order here MUST match the order used in train_model.py — K-Means has no
# concept of column names, only column positions, so a mismatched order silently produces
# wrong predictions instead of an error.
FEATURE_COLUMNS = [
    "listening_hours_per_week",
    "songs_per_day",
    "skip_rate",
    "playlist_count",
]

try:
    model = joblib.load(MODEL_PATH)                    # The trained K-Means model
    scaler = joblib.load(SCALER_PATH)                  # The fitted StandardScaler
    segment_names = joblib.load(SEGMENT_NAMES_PATH)     # {cluster_number: "Casual Listener", ...}
    print("Model, scaler, and segment names loaded successfully.")
except Exception as e:
    # Fail loudly and early with a useful message rather than crashing on the first request
    raise RuntimeError(
        f"Could not load saved files: {e}\n"
        "Run 'python train_model.py' first to create model.pkl, scaler.pkl, and segment_names.pkl."
    )

# Short blurbs shown alongside the predicted segment so the result means something to a
# non-technical reader, not just a cluster label.
SEGMENT_BLURBS = {
    "Casual Listener": "Listens in short, occasional bursts and keeps a small playlist collection.",
    "Music Explorer": "Listens a moderate amount and skips fairly often while sampling new tracks.",
    "Heavy Listener": "Listens for long stretches most days and keeps a large, active playlist collection.",
}


# ------------------------------------------------------------------
# STEP 3: ROUTE - FORM PAGE AND PREDICTION
# ------------------------------------------------------------------
# GET  -> show the empty input form
# POST -> the form was submitted: read the values, predict, show the form again with the result
@app.route("/", methods=["GET", "POST"])
def index():
    result = None    # Populated only after a successful prediction
    error = None      # Populated only if the submitted input was invalid
    form_values = {}  # Echoed back into the form so submitted numbers are not lost on redisplay

    if request.method == "POST":
        try:
            # STEP 3a: Read the four form fields and convert them to numbers.
            # request.form always returns strings, so float() must be applied explicitly.
            listening_hours = float(request.form["listening_hours_per_week"])
            songs_per_day = float(request.form["songs_per_day"])
            skip_rate = float(request.form["skip_rate"])
            playlist_count = float(request.form["playlist_count"])

            form_values = {
                "listening_hours_per_week": listening_hours,
                "songs_per_day": songs_per_day,
                "skip_rate": skip_rate,
                "playlist_count": playlist_count,
            }

            # Basic sanity checks. skip_rate is a percentage, the others cannot be negative.
            if any(v < 0 for v in form_values.values()):
                raise ValueError("Values cannot be negative.")
            if not (0 <= skip_rate <= 100):
                raise ValueError("Skip rate must be between 0 and 100.")

            # STEP 3b: Build a single-row table in the SAME feature order used during training.
            new_listener = [[
                listening_hours,
                songs_per_day,
                skip_rate,
                playlist_count,
            ]]

            # STEP 3c: Scale the new input using the SAVED scaler.
            # transform() only — never fit_transform() here. Fitting again would learn a new
            # mean/standard deviation from a single row, making the scaled value meaningless
            # against cluster centres that were learned on the full training distribution.
            new_listener_scaled = scaler.transform(new_listener)

            # STEP 3d: Predict which of the 3 learned clusters this listener is closest to.
            cluster_number = int(model.predict(new_listener_scaled)[0])

            # STEP 3e: Translate the raw cluster number into the meaningful segment name.
            segment = segment_names[cluster_number]

            result = {
                "cluster_number": cluster_number,
                "segment": segment,
                "blurb": SEGMENT_BLURBS.get(segment, ""),
            }

        except (KeyError, ValueError) as e:
            error = f"Please check your input: {e}"
        except Exception as e:
            error = f"Something went wrong: {e}"

    return render_template("index.html", result=result, error=error, form_values=form_values)


# ------------------------------------------------------------------
# STEP 4: START THE SERVER
# ------------------------------------------------------------------
# This block runs only when the file is executed directly (python app.py),
# not when it is imported by another module.
if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
