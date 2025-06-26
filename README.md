# Car Parking Space Time Tracker

A Computer Vision-based Parking Space Time Tracker that monitors individual parking spaces, tracks vehicles using unique IDs, and logs their entry and exit times. The system visually indicates the status of each parking spot — **green** for available and **red** for occupied.

## Features

*  Detects and tracks vehicles in each parking space.
*  Logs **entry** and **exit** times for each vehicle.
*  Assigns a **unique Track ID** to every detected vehicle.
*  Visual feedback:

  * **Green** — parking space available.
  * **Red** — parking space occupied.
*  Generates log files with:

  * Vehicle ID
  * Station (parking space) number
  * Entry time
  * Exit time


## Getting Started

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the application:

```bash
python main.py
```

## Log Format

The system logs the following details in `parking_log.csv`:

| Vehicle ID | Station Number | Entry Time          | Exit Time           |
| ---------- | -------------- | ------------------- | ------------------- |
| 101        | 2              | 2025-06-26 10:15:23 | 2025-06-26 11:05:48 |
