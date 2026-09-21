# ------------------------------------------------------------------
# TRAINING SCRIPT: train_model.py
#
# Run this ONCE (or whenever the dataset changes):
#     python train_model.py
#
# It reads listener behaviour data, scales it, groups listeners into
# 3 clusters with K-Means, prints what each cluster looks like, and
# saves the trained model and scaler to disk so the app never has to
# retrain.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# MODULE IMPORTS
# ------------------------------------------------------------------
import os                                     # Builds OS-independent file paths
import pandas as pd                           # Data manipulation library for loading the CSV into a DataFrame
import joblib                                 # Utility for saving (serializing) trained objects to .pkl files

from sklearn.preprocessing import StandardScaler  # Rescales features so no single column dominates distance
from sklearn.cluster import KMeans                # Unsupervised clustering algorithm


# ------------------------------------------------------------------
# STEP 0: BUILD ABSOLUTE FILE PATHS
# ------------------------------------------------------------------
# __file__ is this script. os.path.dirname() gives the folder it lives in.
# Building paths this way means the script works no matter which folder you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "music_listeners.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


# ------------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------------------------------
# Read the CSV of listener behaviour into a pandas DataFrame.
# Unlike a supervised project, there is no label column here — nobody has told us in
# advance which listeners are "Casual", "Explorer" or "Heavy". Discovering those groups
# is exactly what the clustering model is for.
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna()  # Remove any incomplete rows so they cannot distort the clustering

print(f"Dataset loaded: {len(df)} listeners")


# ------------------------------------------------------------------
# STEP 2: SELECT THE LISTENER BEHAVIOUR FEATURES
# ------------------------------------------------------------------
# X (features) -> the four numeric columns that describe listening behaviour.
# There is no y (target) in this project — that is what makes it unsupervised.
FEATURE_COLUMNS = [
    "listening_hours_per_week",
    "songs_per_day",
    "skip_rate",
    "playlist_count",
]
X = df[FEATURE_COLUMNS]

print(f"Features used: {FEATURE_COLUMNS}")


# ------------------------------------------------------------------
# STEP 3: SCALE THE FEATURES
# ------------------------------------------------------------------
# K-Means groups points by distance, so a column measured in bigger numbers (songs_per_day,
# which can reach 145) would silently dominate a column measured in smaller numbers
# (playlist_count, which tops out around 40) unless every column is put on the same scale first.
# StandardScaler rescales each column to have mean 0 and standard deviation 1.
scaler = StandardScaler()

# .fit_transform() on the TRAINING data: learn the mean/standard deviation of each column
# AND apply that scaling in one step.
X_scaled = scaler.fit_transform(X)

print("Features scaled with StandardScaler.")


# ------------------------------------------------------------------
# STEP 4: CREATE THE K-MEANS MODEL
# ------------------------------------------------------------------
# n_clusters=3 -> the brief asks for exactly three listener groups
# random_state=42 -> makes the cluster assignment reproducible run to run
# n_init=10 -> K-Means starts from 10 different random centre positions and keeps the
#              best result, since a single random start can land on a poor grouping
model = KMeans(n_clusters=3, random_state=42, n_init=10)


# ------------------------------------------------------------------
# STEP 5: TRAIN THE MODEL
# ------------------------------------------------------------------
# .fit() is the learning step. K-Means repeatedly: (1) assigns each listener to its
# nearest of the 3 centres, then (2) moves each centre to the average of the listeners
# now assigned to it, until the centres stop moving.
model.fit(X_scaled)

print("Model training complete.")


# ------------------------------------------------------------------
# STEP 6: FIND AND UNDERSTAND THE THREE LISTENER GROUPS
# ------------------------------------------------------------------
# K-Means only ever outputs cluster numbers (0, 1, 2) — it has no idea what a "Casual
# Listener" is. Turning those numbers into meaningful names means looking at each
# cluster's centre and reading its behaviour back out in original (unscaled) units.
#
# inverse_transform() undoes the StandardScaler math, converting the cluster centres
# from scaled units back into real hours/day/percent/count values that a human can read.
centres_original_units = scaler.inverse_transform(model.cluster_centers_)
centres_df = pd.DataFrame(centres_original_units, columns=FEATURE_COLUMNS)
centres_df.index.name = "cluster"

print("\n--------------------------------------------------")
print("CLUSTER CENTRES (original units)")
print("--------------------------------------------------")
print(centres_df.round(2))

# Rank the 3 clusters by overall listening activity (hours/week is the clearest single
# signal) so the lowest-activity cluster becomes "Casual Listener" and the highest
# becomes "Heavy Listener", exactly as the brief describes.
activity_rank = centres_df["listening_hours_per_week"].sort_values().index.tolist()
SEGMENT_NAMES = {
    activity_rank[0]: "Casual Listener",
    activity_rank[1]: "Music Explorer",
    activity_rank[2]: "Heavy Listener",
}

print("\nCluster number -> Listener segment (by listening activity):")
for cluster_num, name in SEGMENT_NAMES.items():
    print(f"  Cluster {cluster_num}  ->  {name}")

# Show how many listeners fell into each group in the training data
df["cluster"] = model.labels_
print("\nListeners per segment:")
for cluster_num, name in SEGMENT_NAMES.items():
    count = (df["cluster"] == cluster_num).sum()
    print(f"  {name:<16} ({cluster_num}): {count} listeners")


# ------------------------------------------------------------------
# STEP 7 & 8: SAVE THE TRAINED MODEL AND SCALER
# ------------------------------------------------------------------
# Two objects must be saved together, not just the model:
#   1. the K-Means model -> holds the 3 learned cluster centres
#   2. the StandardScaler -> holds the mean/standard deviation learned from training
# If the scaler were refit later, or skipped at prediction time, a new listener would be
# measured on a different scale than the one the cluster centres were learned on, and
# every prediction would be meaningless.
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

# The segment name mapping is saved too, so app.py does not have to re-derive
# "which cluster number means Casual" every time it starts up.
SEGMENT_NAMES_PATH = os.path.join(BASE_DIR, "segment_names.pkl")
joblib.dump(SEGMENT_NAMES, SEGMENT_NAMES_PATH)

print("\nSaved successfully:")
print(f"  {MODEL_PATH}")
print(f"  {SCALER_PATH}")
print(f"  {SEGMENT_NAMES_PATH}")
print("\nYou can now run the application with:  python app.py")
