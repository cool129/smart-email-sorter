import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

data = pd.read_csv("emails.csv")

data["text"] = data["Subject"] + " " + data["Body"]

X = data["text"]
y = data["Priority"]

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X, y)

print("✅ Smart Email Sorter is ready!")
print("Type 'exit' to quit.\n")

while True:
    subject = input("Enter Email Subject: ")
    if subject.lower() == "exit":
        break

    body = input("Enter Email Body: ")
    if body.lower() == "exit":
        break

    email_text = [subject + " " + body]

    prediction = model.predict(email_text)

    print("Predicted Priority:", prediction[0])
    print("---")