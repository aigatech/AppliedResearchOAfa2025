from mailReader import get_last_emails
from calendarTest import make_calendar_event
from classifier import classify_text
from inferenceModel import get_model_inference, MultiTokenStoppingCriteria

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import re
import datetime as dt

def main():
    emails = get_last_emails(1)

    for email in emails:
        if classify_text(email) in ["Meeting", "Event", "Workshop"]:
            print("Event detected")
            event_params = get_model_inference(email)
            event_params = event_params.split(";")
            # print(event_params)
            if event_params[1] != "N/A":
                # Pre-processing of inputs
                equalsSign = event_params[1].find("=")
                event_params[1] = event_params[1][equalsSign + 1:].strip()
                event_params[1] = event_params[1].replace("/", " ").split(" ")
                #print(event_params[1])
                event_params[1] = dt.datetime(2025, int(event_params[1][0]), int(event_params[1][1]), int(event_params[1][2]), int(event_params[1][3]))
                
                equalsSign = event_params[0].find("=")
                event_params[0] = event_params[0][equalsSign + 1:].strip()

                equalsSign = event_params[2].find("=")
                event_params[2] = event_params[2][equalsSign + 1:].strip()

                #print(event_params[0])
                #print(event_params[1])
                #print(event_params[2])

                make_calendar_event(event_params[0], event_params[2], event_params[1])
                print("Calendar event created")


if __name__ == "__main__":
    main()
