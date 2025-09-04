import datetime as dt
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import requests

SCOPES = ["https://www.googleapis.com/auth/calendar"]

def make_calendar_event(summary, location, start_time):
    creds = None
    if os.path.exists("token1.json"):
        creds = Credentials.from_authorized_user_file("token1.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("token1.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("calendar", "v3", credentials=creds)
        event = {
            "summary":summary,
            "location": location,
            "description": "",
            "colorId": 9,
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "America/New_York"
            },
            "end": {
                "dateTime": (start_time + dt.timedelta(hours=1)).isoformat(), # 1 hour long
                "timeZone": "America/New_York"
            }
        }
        event = service.events().insert(calendarId="primary", body=event).execute()
        print(f'Event created: {event.get("htmlLink")}')

    except HttpError as error:
        print(f"An error occurred: {error}")

'''
if __name__ == "__main__":
    make_calendar_event( "Python-generated event", "online, duh", dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))))
'''












'''
            "recurrence": {
                "frequency": "DAILY",
                "interval": 1
            },
            "attendees": {
                {"email": "arthur.armaing@gmail.com"}
            }
'''


'''
        # to list the last 5 events
        now = dt.datetime.now(dt.timezone.utc).isoformat()

        event_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=5,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        events = event_result.get("items", [])

        if not events:
            print("No upcoming events found")
            return
        
        for event in events:
            start = event["start"].get("dateTime") # .get("dateTime", event["start"].get("date"))
            print(start, " ", event["summary"])
            print(type(start))
'''