EmailToCalendar:

Takes in the last n emails, and scans each email for possible events to add to your calendar. If it detects events, it will connect to your google calendar and add them to it, giving time and location for each event as well as a short description in the title. 
This is all provided that you have proper credentials to enter your google calendar and a Cloud project.

HENCE, for the sake of running the project, I will be leaving my calendar credentials to this project. THAT SAID, PLEASE DO NOT ABUSE THE CREDENTIALS in ANY WAY to modify, delete, or send calendar events or emails.

The project can be run by running main.py. The default is set to 2 email, but can be changed to more if needed. In the example I will be providing, I will be using 1 email, containing a general body meeting on tuesday at 6:30pm.

Please install the dependencies such as the transformers library, Python, pyTorch, and the provided libraries imported in each file.

The project can be expanded to schedule the task of running the script, and with accuracy of the LLM, both of which I intend to do on my free time.