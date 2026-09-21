# Email Spam Detection — Supervised Machine Learning

Classifies an email as **Spam** or **Not Spam** using a supervised classifier
trained on labelled examples. TF-IDF turns the text into numbers, Multinomial
Naive Bayes makes the decision, FastAPI serves it, and a plain HTML page is the
interface.

---

## Project structure

```
email_spam_detection/
├── requirements.txt          Dependency manifest
├── README.md                 This file
├── backend/
│   ├── train_model.py        Training script: loads data, trains, evaluates, saves
│   ├── app.py                FastAPI app: loads the saved model, serves /predict
│   ├── spam_emails.csv       Labelled dataset (40 emails, columns: email, label)
│   ├── spam_model.pkl        Saved classifier          ← created by train_model.py
│   └── vectorizer.pkl        Saved TF-IDF vectorizer   ← created by train_model.py
└── frontend/
    ├── index.html            Page structure
    ├── style.css             Styling
    └── script.js             Sends the text to the API, renders the verdict
```

The architecture document lists the saved model twice (`model.pkl` and
`spam_model.pkl`). Only one classifier file is needed, so it is `spam_model.pkl`.
The second artifact is `vectorizer.pkl` — the fitted text preprocessor, which
**must** be saved alongside the model so prediction uses exactly the same
word-to-column mapping as training.

---

## Setup and run

### 1. Open the project

Open the `email_spam_detection` folder in VS Code
(**File → Open Folder…**), then open a terminal with **Terminal → New Terminal**.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Train the model

```bash
cd backend
python train_model.py
```

This prints the accuracy and classification report, then writes `spam_model.pkl`
and `vectorizer.pkl`. Both files are already included, so you can skip this step
and run it again only if you change the dataset.

### 4. Start the API

```bash
python app.py
```

Leave this terminal running. The API is now at `http://127.0.0.1:8000`, and the
interactive docs are at `http://127.0.0.1:8000/docs`.

### 5. Open the frontend

Open `frontend/index.html` in your browser — double-clicking the file works.
Paste a message, or load one of the two examples, and press **Check this
message**.

---

## How the model works

| Stage | What happens |
|---|---|
| Features (X) | The raw email text |
| Target (y) | The label: `1` = spam, `0` = not spam |
| Preprocessing | TF-IDF converts text into numeric vectors, weighting rare informative words above common ones |
| Split | 80% train / 20% test, stratified so both halves keep the same spam ratio |
| Model | Multinomial Naive Bayes, the standard algorithm for text classification |
| Evaluation | Accuracy, precision, recall and a confusion matrix, all on the unseen test set |
| Saving | Classifier and vectorizer are pickled so the API never retrains |

### Flow at prediction time

```
new email → vectorizer.transform() → model.predict() → Spam / Not Spam
```

Note `.transform()`, not `.fit_transform()`. Refitting on a new email would
rebuild the vocabulary and invalidate everything the model learned.

---

## API reference

**POST** `/predict`

Request:

```json
{ "email_text": "Win a free iPhone now! Click this link to claim your prize." }
```

Response:

```json
{
  "prediction": 1,
  "label": "Spam",
  "confidence": 96.4,
  "spam_probability": 96.4,
  "email_text": "Win a free iPhone now! Click this link to claim your prize."
}
```

**GET** `/` returns a health check so you can confirm the server is up.

---

## Troubleshooting

**"Could not load model files"** — run `python train_model.py` from inside the
`backend` folder first.

**"Cannot reach the API"** in the browser — the backend is not running, or not
on port 8000. Check the terminal from step 4.

**Predictions feel shaky** — the dataset is only 40 emails, which is enough to
demonstrate the pipeline but not enough to generalise well. Add more labelled
rows to `spam_emails.csv` and retrain. Test accuracy on 8 held-out emails is a
small sample, so treat a perfect score as a sign the test set is tiny rather
than proof the model is flawless.

---

## Concepts demonstrated

Labelled data · features and targets · text vectorisation · train/test split ·
model training · evaluation on unseen data · model persistence · serving
predictions through an API
