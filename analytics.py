# Importing modules
from ultralytics import YOLO
import numpy as np
import cv2
import math
from datetime import datetime
from datetime import timedelta
import os

class detections():

    def __init__(self, video_path, roi,entry_timestamps, exit_timestamps, log_file_path, model):
        self.video = video_path
        self.roi = roi
        self.model = model
        self.entry_time = entry_timestamps 
        self.exit_time = exit_timestamps
        self.log_path = log_file_path

    # logging function
    # creates a txt file and logs in data
    def logging(self, entry, exit, csv_path):

        # time data for logging file name
        file_create_time = datetime.now()
        file_create_time = f"{file_create_time.hour}_{file_create_time.minute}_{file_create_time.second}"

        # defining the logging file path
        txt_filename = f"logging_{file_create_time}.txt"
        txt_file_path = os.path.join(csv_path, txt_filename)

        # creating and logging in data
        with open(txt_file_path, 'a') as f:

            # when file is empty log in the intial data
            if os.path.getsize(txt_file_path) == 0: 
                
                for key, val in entry.items():
                    f.write(f"Station number: {key} Entry time: {val}\n")
                    
            # once file has data in it logging in the latest entries
            elif os.path.getsize(txt_file_path) > 0:
                
                entry_items, exit_items = list(entry.items()), list(exit.items())

                if len(entry_items) > 0:
                    latest_entry = entry_items[-1]
                    f.write(f"Station number: {latest_entry[0]} Entry time: {latest_entry[1]}\n")

                if len(exit_items) > 0:
                    latest_exit = exit_items[-1]
                    f.write(f"Station number: {latest_exit[0]} Exit time: {latest_exit[1]}\n")

            f.close()

    # the timer function to keep track of time for each ROI
    def timer(self, entry_time, current_time):

        # time formatting
        start_time = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")

        time_diff = end_time - start_time

        sec = time_diff.total_seconds()

        time = str(timedelta(seconds= sec))

        return time

    # checking the distance to identify the occupancy of the ROI
    def check_dist(self, parked_points: list, center_corr: tuple):
            
        points_dist = []

        # measuring distance between the car and ROI
        for point in parked_points:
            dist = math.dist(point, center_corr)
            points_dist.append(dist)
            dist = min(points_dist)

        points_dist.clear()
        
        return dist

    def detect(self, model, frame):
        
        white = (255, 255, 255)

        results = model(frame) # performing detection
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32) # detected bounding boxes
        classes = results[0].boxes.cls.cpu().numpy().astype(np.int32) # class represented by integer
        classes_name = results[0].names # class dictionary

        tracking = ['car']

        car_centre = []

        video_timestamp = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        cv2.putText(frame, video_timestamp, (700, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, white, 2) # Putting time stamps on data
    
        # calculting the centroid of the detections
        for bbox, cls in zip(bboxes, classes):

            label = classes_name[cls]

            if label in tracking:
            
                x0, y0, x1, y1 = map(int, bbox)
                
                bbox_center = (int((x0+x1)/2), int((y0+y1)/2))

                car_centre.append(bbox_center)

        return car_centre
    
    def draw_roi(self, roi, frame):

        red = (0, 0, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)

        current_time = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) # time values in real time

        # drawing the ROIs
        for key, val in roi.items():
            
            x0, y0, x1, y1 = val[0][0], val[0][1], val[2][0], val[2][1] # mapping the co-ordinates
            pt1, pt2 = (x0, y0), (x1, y1) # defining the points

            car_center = self.detect(self.model, frame)

            dist = self.check_dist(car_center, val[4])

            dist = int(dist) # calculating distance between detected and ROI centroids

            # in case the ROI is occupied the color will be red
            # storing and deleting timestamps based on occupation history
            if dist <= 40:
                color = red

                # saving the entry timestamp and dropping the previous exit timestamp
                if bool(self.entry_time.get(key)) == False:
                    self.entry_time[key] = current_time
                    
                    if bool(self.exit_time.get(key)) == True:
                        del self.exit_time[key]
            
                entry_time = self.entry_time[key]

            else:
                color = green

            # drawing ROI and station IDs on feed
            cv2.rectangle(frame, pt1, pt2, color, 1)
            cv2.putText(frame, key, (x1-40, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                yellow, 1)
            
            if color == red:
                # writing time on feed if the ROI is occupied
                time = self.timer(entry_time, current_time)
                cv2.putText(frame, time, (x0, y0 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    yellow, 1)

        return frame

    def run_video(self):
        cap = cv2.VideoCapture(self.video) # reading the video

        run = True
        frame_count = 0
        entry_prev_len, exit_prev_len = 0,0

        # running the feed
        while run:

            ret, frame = cap.read()

            if ret:
                
                # bbox_center = self.detect(self.model, frame)
                frames = f"Frames: {frame_count}"

                frame - self.draw_roi(self.roi, frame)

                # logging timestamps in case the entry or exit timestamps are updated
                if len(self.entry_time) > entry_prev_len or len(self.exit_time) > exit_prev_len:
                    self.logging(self.entry_time, exit, self.log_path)

                entry_prev_len, exit_prev_len = len(self.entry_time), len(self.exit_time) # updating the number of timestamps

                print(entry_prev_len, exit_prev_len)

                cv2.putText(frame, frames, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 2)
                cv2.imshow('img', frame) # runnig the feed

                # print(bbox_center)

                frame_count += 1

            # press 'q' to quit
            k = cv2.waitKey(1)

            if k == ord('q') or not ret:
                run = False

        cap.release()
        cv2.destroyAllWindows()

