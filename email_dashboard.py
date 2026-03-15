import pandas as pd
from tkinter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load dataset
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

# Counters
high_count = 0
medium_count = 0
low_count = 0

# GUI window
window = Tk()
window.title("Smart Email Dashboard")
window.geometry("500x400")

Label(window, text="Smart Email Sorter Dashboard", font=("Arial", 16)).pack(pady=10)

# Input fields
Label(window, text="Email Subject").pack()
subject_entry = Entry(window, width=60)
subject_entry.pack()

Label(window, text="Email Body").pack()
body_entry = Entry(window, width=60)
body_entry.pack()

# Email list display
email_list = Listbox(window, width=70)
email_list.pack(pady=10)

# Counter labels
counter_label = Label(window, text="High: 0 | Medium: 0 | Low: 0", font=("Arial", 12))
counter_label.pack(pady=10)

# Prediction function
def predict_priority():
    global high_count, medium_count, low_count

    subject = subject_entry.get()
    body = body_entry.get()

    text = [subject + " " + body]
    prediction = model.predict(text)[0]

    if prediction == "High":
        high_count += 1
    elif prediction == "Medium":
        medium_count += 1
    else:
        low_count += 1

    email_list.insert(END, f"{subject} → {prediction}")

    counter_label.config(
        text=f"High: {high_count} | Medium: {medium_count} | Low: {low_count}"
    )

# Button
Button(window, text="Analyze Email", command=predict_priority).pack(pady=5)

window.mainloop()