# Push-up Unlocker

## What it Does
Push-up Unlocker is a fun productivity tool that encourages exercise by giving you timed access to distracting websites. For every push-up you complete, you earn an equivalent number of minutes to access websites like YouTube or Instagram. For example, doing 10 push-ups unlocks 10 minutes of scrolling time.

This project uses your webcam to track push-ups in real-time using **MediaPipe** and **OpenCV**. The program automatically blocks and unblocks websites by modifying your system's hosts file.

> ⚠️ **Important:** This program only works if you run it with **administrator privileges**, as it needs permission to modify your system hosts file.

## How to Run
1. **Install dependencies**  
   Make sure you have Python installed (Python 3.8+ recommended). Then install the required packages:
   ```bash
   pip install opencv-python mediapipe numpy
2. **Run the script as Administrator**
  On Windows:
  Right-click your terminal or IDE and select Run as Administrator.

   Execute the Python script: python app3.py
3. **Perform push-ups in front of your webcam**

   The program will detect your push-ups and keep count.
   Once you reach the minimum number of push-ups (default 5), websites will be unblocked for a number of minutes equal to the push-up count.

4. **Enjoy your unlocked websites!**
After the unlocked time ends, websites will be automatically blocked again. Repeat push-ups to unlock more time.

## Notes

- Tested on Windows. Hosts file paths may differ for Mac or Linux.
- Ensure your webcam is properly connected.
- Push-up detection is angle-based and may not detect perfect form. 
