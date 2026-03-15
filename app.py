from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load dataset
data = pd.read_csv("emails.csv")

# Combine subject and body
data["text"] = data["Subject"] + " " + data["Body"]

X = data["text"]
y = data["Priority"]

# Machine learning model
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X, y)

history = []

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    spam = None
    confidence = None
    bulk = None

    high = medium = low = 0

    if request.method == "POST":

        # SINGLE EMAIL ANALYSIS
        subject = request.form.get("subject")
        body = request.form.get("body")

        if subject:

            text = subject + " " + body

            prediction = model.predict([text])[0]

            confidence = round(max(model.predict_proba([text])[0]) * 100, 2)

            if "free" in text.lower() or "winner" in text.lower():
                spam = "Yes"
            else:
                spam = "No"

            history.append({
                "subject": subject,
                "priority": prediction,
                "spam": spam
            })

        # CSV UPLOAD
        file = request.files.get("file")

        if file:

            df = pd.read_csv(file)

            df["text"] = df["Subject"] + " " + df["Body"]

            predictions = model.predict(df["text"])

            bulk = []

            for subject, priority in zip(df["Subject"], predictions):

                bulk.append({
                    "subject": subject,
                    "priority": priority
                })

    # Count priorities for chart
    for email in history:

        if email["priority"] == "High":
            high += 1
        elif email["priority"] == "Medium":
            medium += 1
        else:
            low += 1

    return render_template(
        "index.html",
        prediction=prediction,
        spam=spam,
        confidence=confidence,
        history=history,
        bulk=bulk,
        high=high,
        medium=medium,
        low=low
    )


if __name__ == "__main__":
    app.run(debug=True)