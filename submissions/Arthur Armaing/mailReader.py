
import os.path
import datetime as dt
import base64

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_last_emails(num_emails):
  creds = None
  
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    
    with open("token.json", "w") as token:
      token.write(creds.to_json())
      
  message_output = []
  try:
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    service = build("gmail", "v1", credentials=creds)
    results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=num_emails
        ).execute()
    messages = results.get("messages", [])

    if not messages:
        print("No messages found.")
        return

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()

        '''headers = msg_data["payload"]["headers"]
        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown Sender")
        date = next((h["value"] for h in headers if h["name"] == "Date"), "Unknown Date")'''

        # Try to extract body text
        body = ""
        if "data" in msg_data["payload"]["body"]:  # Simple email
            body = msg_data["payload"]["body"]["data"]
        elif "parts" in msg_data["payload"]:  # Multipart email
            for part in msg_data["payload"]["parts"]:
                if part["mimeType"] == "text/plain" and "data" in part["body"]:
                    body = part["body"]["data"]
                    break
        if body:
            body = base64.urlsafe_b64decode(body).decode("utf-8", errors="ignore")
            message_output.append(body[:700])
        else:
            body = "[No text body found]"
        
        '''print(f"From: {sender}")
        print(f"Subject: {subject}")
        print(f"Date: {date}")
        print(f"Body:\n{body[:500]}") 
        print("-" * 40)'''

  except HttpError as error:
    print(f"An error occurred: {error}")

  return message_output


'''
if __name__ == "__main__":
  out = get_last_5_emails()
  print(out)
'''
