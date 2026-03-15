import pandas as pd
from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load training data
data = pd.read_csv("emails.csv")

data["text"] = data["Subject"] + " " + data["Body"]

X = data["text"]
y = data["Priority"]

# Train model
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X, y)

# GUI Window
window = Tk()
window.title("Smart Email Sorter")
window.geometry("400x300")

# Subject Label
Label(window, text="Email Subject").pack()

subject_entry = Entry(window, width=50)
subject_entry.pack()

# Body Label
Label(window, text="Email Body").pack()

body_entry = Entry(window, width=50)
body_entry.pack()

# Result Label
result_label = Label(window, text="Priority will appear here")
result_label.pack(pady=20)

# Function to predict
def predict_priority():
    subject = subject_entry.get()
    body = body_entry.get()

    text = [subject + " " + body]

    prediction = model.predict(text)

    result_label.config(text="Predicted Priority: " + prediction[0])

# Button
Button(window, text="Check Priority", command=predict_priority).pack()

window.mainloop()