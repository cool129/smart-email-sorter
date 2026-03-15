import os
import pickle
import base64
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --------------------------
# 1️⃣ Gmail API Setup
# --------------------------
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('gmail', 'v1', credentials=creds)

# --------------------------
# 2️⃣ Fetch Latest Emails
# --------------------------
results = service.users().messages().list(userId='me', maxResults=10).execute()
messages = results.get('messages', [])

emails = []
for msg in messages:
    txt = service.users().messages().get(userId='me', id=msg['id']).execute()
    payload = txt['payload']
    headers = payload.get('headers', [])
    subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
    body = ''
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body = base64.urlsafe_b64decode(part['body']['data']).decode()
    emails.append({'Subject': subject, 'Body': body})

# --------------------------
# 3️⃣ Load Email Dataset & Train Model
# --------------------------
data = pd.read_csv("emails.csv")  # your CSV with Subject,Body,Priority
data["text"] = data["Subject"] + " " + data["Body"]
X = data["text"]
y = data["Priority"]

model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])
model.fit(X, y)

print("✅ Gmail AI Email Sorter is ready!\n")

# --------------------------
# 4️⃣ Predict Priorities
# --------------------------
for email in emails:
    text = [email['Subject'] + " " + email['Body']]
    priority = model.predict(text)[0]
    print(f"Email: {email['Subject']}")
    print(f"Predicted Priority: {priority}")
    print("---")