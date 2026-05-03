# 📧 Smart Email Sorter

Smart Email Sorter is an AI-powered email analysis tool that automatically classifies emails by priority (High, Medium, Low) and detects spam using machine learning.

The project is built with:

- Python
- Flask
- Scikit-learn
- Chart.js
- Bootstrap

---

# Features

- AI email priority classification
- Spam detection
- Confidence score
- CSV bulk email analyzer
- Email history dashboard
- Email priority charts
- Dark mode UI
- Smart search filter

---

# Installation

Clone the repository:

git clone https://github.com/cool129/smart-email-sorter.git

Move into the project folder:

cd smart-email-sorter

Install the required dependencies:

pip install -r requirements.txt

---

# Running the Project

Start the Flask server:

python app.py

Then open your browser and go to:

http://127.0.0.1:5000

You should see the **Smart Email AI Dashboard**.

---

# Project Structure

smart-email-sorter,
app.py,
requirements.txt,
templates,
index.html,
README.md

---

# Deployment

This project is deployed using Render.

Live demo:
https://smart-email-sorter-wy88.onrender.com

Here's the prompt to put in
Subject	Body	Priority	Spam
Urgent server issue	Server is down please fix	High	No
Meeting reminder	Team meeting tomorrow	Medium	No
Weekly newsletter	Updates for this week	Low	No
Password reset required	Reset password immediately	High	No
Win a free iPhone	Click here to claim prize	Low	Yes
Crypto investment	double your money fast	Low	Yes
Meeting reminder	Team meeting tomorrow	Medium	No
