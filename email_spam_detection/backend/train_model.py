# ------------------------------------------------------------------
# TRAINING SCRIPT: train_model.py
#
# Run this ONCE (or whenever the dataset changes):
#     python train_model.py
#
# It reads the labelled dataset, converts the email text into numbers,
# trains a spam/ham classifier, evaluates it, and saves the trained
# artifacts to disk so the API never has to retrain.
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# MODULE IMPORTS
# ------------------------------------------------------------------
import os                                                   # Used to build file paths that work on any operating system
import pandas as pd                                         # Data manipulation library for loading the CSV into a DataFrame
import joblib                                               # Utility for saving (serializing) trained objects to .pkl files

from sklearn.model_selection import train_test_split        # Splits the dataset into training and testing portions
from sklearn.feature_extraction.text import TfidfVectorizer # Converts raw email text into numerical feature vectors
from sklearn.naive_bayes import MultinomialNB               # Supervised classification algorithm well suited to text counts
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation metrics


# ------------------------------------------------------------------
# STEP 0: BUILD ABSOLUTE FILE PATHS
# ------------------------------------------------------------------
# __file__ is this script. os.path.dirname() gives the folder it lives in (backend/).
# Building paths this way means the script works no matter which folder you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "spam_emails.csv")       # Input: the labelled dataset
MODEL_PATH = os.path.join(BASE_DIR, "spam_model.pkl")       # Output: the trained classifier
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")  # Output: the fitted text preprocessing object


# ------------------------------------------------------------------
# STEP 1: LOAD THE LABELLED DATASET
# ------------------------------------------------------------------
# Read the CSV of example emails that a human has already marked as spam (1) or not spam (0).
# Supervised learning requires these labels — that is what the model learns from.
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Remove any rows with missing values so they cannot corrupt training
df = df.dropna(subset=["email", "label"])

print(f"Dataset loaded: {len(df)} labelled emails")
print(f"  Spam (1)     : {(df['label'] == 1).sum()}")
print(f"  Not spam (0) : {(df['label'] == 0).sum()}")


# ------------------------------------------------------------------
# STEP 2: SEPARATE FEATURES (X) AND TARGET (y)
# ------------------------------------------------------------------
# X (features)  -> the raw email text, the input the model looks at
# y (target)    -> the known spam/ham label, the answer the model tries to reproduce
X = df["email"].astype(str)
y = df["label"].astype(int)


# ------------------------------------------------------------------
# STEP 3: SPLIT THE DATA INTO TRAIN AND TEST SETS
# ------------------------------------------------------------------
# The model must be judged on emails it has never seen, otherwise a model that simply
# memorised the training set would look perfect while being useless in practice.
#   test_size=0.2   -> keep 20% of the rows aside purely for evaluation
#   random_state=42 -> makes the split identical every run (reproducible results)
#   stratify=y      -> keeps the spam/ham ratio the same in both halves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining emails: {len(X_train)}  |  Testing emails: {len(X_test)}")


# ------------------------------------------------------------------
# STEP 4: CONVERT TEXT INTO NUMERICAL FEATURES (TF-IDF)
# ------------------------------------------------------------------
# Machine learning models cannot read words — they need numbers.
# TF-IDF (Term Frequency - Inverse Document Frequency) turns each email into a vector where:
#   - words that appear often in THIS email score higher (term frequency)
#   - words that appear in EVERY email score lower (inverse document frequency)
# So common filler like "the" is pushed down, while giveaway words like "winner" stand out.
#
#   lowercase=True    -> "FREE" and "free" are treated as the same word
#   stop_words        -> drops uninformative English words such as "the", "is", "and"
#   ngram_range=(1,2) -> learns single words AND two-word phrases like "click here"
#   min_df=1          -> keep every term (the dataset here is small)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

# .fit_transform() on TRAINING data only:
#   fit      -> learn the vocabulary and the IDF weights
#   transform-> convert the training emails into a numeric matrix
X_train_features = vectorizer.fit_transform(X_train)

# .transform() on TEST data:
# We deliberately do NOT call fit here. The test set must be converted using the vocabulary
# learned from the training set, exactly as a brand new email will be at prediction time.
X_test_features = vectorizer.transform(X_test)

print(f"Vocabulary size learned: {len(vectorizer.vocabulary_)} terms")


# ------------------------------------------------------------------
# STEP 5: TRAIN THE SUPERVISED CLASSIFIER
# ------------------------------------------------------------------
# Multinomial Naive Bayes is the classic algorithm for text classification.
# It learns, for every word, how much more likely that word is to appear in spam than in ham,
# then multiplies those probabilities together to score a whole message.
#   alpha=0.5 -> smoothing, so a word never seen during training does not force a probability of zero
model = MultinomialNB(alpha=0.5)

# .fit() is the actual learning step: map input features (X_train_features) to labels (y_train)
model.fit(X_train_features, y_train)

print("Model training complete.")


# ------------------------------------------------------------------
# STEP 6: EVALUATE THE MODEL ON UNSEEN TEST DATA
# ------------------------------------------------------------------
# Ask the trained model to classify the held-out test emails, then compare its answers
# against the true labels we already know.
y_pred = model.predict(X_test_features)

print("\n--------------------------------------------------")
print("MODEL EVALUATION (on unseen test data)")
print("--------------------------------------------------")

# Accuracy = fraction of test emails classified correctly
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Precision / recall / F1 broken down per class.
# Recall on the spam class matters most: it answers "of all real spam, how much did we catch?"
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"], zero_division=0))

# Confusion matrix rows are the true label, columns are the prediction:
#   [[true ham  , ham wrongly flagged as spam],
#    [spam missed, spam correctly caught     ]]
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# ------------------------------------------------------------------
# STEP 7: SAVE THE TRAINED ARTIFACTS
# ------------------------------------------------------------------
# Two objects must be saved together, not just the model:
#   1. the classifier      -> holds the learned spam/ham probabilities
#   2. the TF-IDF vectorizer -> holds the vocabulary and IDF weights
# If the vectorizer were rebuilt later, word-to-column positions would change and the
# saved model's learned weights would point at the wrong words. Saving both keeps
# training-time and prediction-time preprocessing perfectly identical.
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("\nSaved successfully:")
print(f"  {MODEL_PATH}")
print(f"  {VECTORIZER_PATH}")
print("\nYou can now run the API with:  python app.py")
