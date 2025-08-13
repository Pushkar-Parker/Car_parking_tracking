import cv2
import supervision as sv
from ultralytics import YOLO
import math
from datetime import datetime, timedelta
from collections import deque
import csv
import os

class VideoAnalytics():

    red = (0, 0, 255)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    blue = (255, 0, 0)
    white = (255,255,255)

    session_log = {}

    occupied_station_status = deque(maxlen= 2)

    time_fmt = "%Y-%m-%d %H:%M:%S"

    ROI = {
    'ID_1': [[57, 248], [54, 339], [178, 338], [168, 245], (114, 292)],
    'ID_2': [[194, 263], [198, 339], [304, 339], [287, 262], (245, 300)],
    'ID_3': [[301, 254], [305, 338], [424, 336], [411, 256], (360, 296)],
    'ID_4': [[426, 252], [429, 337], [570, 336], [557, 260], (495, 296)],
    'ID_5': [[528, 206], [532, 327], [689, 332], [666, 207], (603, 268)],
    'ID_6': [[633, 226], [640, 306], [781, 314], [764, 232], (704, 269)],
    'ID_7': [[740, 222], [749, 312], [897, 312], [888, 227], (818, 268)],
    'ID_8': [[13, 151], [12, 213], [82, 210], [82, 151], (47, 181)],
    'ID_9': [[93, 151], [96, 217], [165, 218], [165, 149], (129, 183)],
    'ID_10': [[186, 152], [190, 210], [254, 206], [247, 148], (219, 179)],
    'ID_11': [[261, 144], [263, 205], [336, 207], [331, 136], (297, 173)],
    'ID_12': [[490, 127], [499, 199], [595, 195], [585, 130], (542, 162)],
    'ID_13': [[553, 121], [568, 181], [650, 184], [633, 123], (601, 152)],
    'ID_14': [[620, 115], [627, 181], [715, 178], [703, 111], (666, 146)],
    'ID_15': [[738, 114], [747, 176], [843, 173], [834, 110], (790, 143)]
    }

    # ROI = {
    #     'ID_1': [[190, 395], [189, 462], [276, 464], [278, 388], (233, 427)],
    #     'ID_2': [[289, 392], [290, 454], [359, 462], [362, 394], (325, 425)],
    #     'ID_3': [[373, 399], [371, 463], [441, 468], [438, 401], (405, 432)],
    #     'ID_4': [[460, 401], [459, 467], [533, 462], [521, 400], (493, 432)],
    #     'ID_5': [[534, 398], [532, 461], [607, 463], [600, 400], (568, 430)],
    #     'ID_6': [[605, 374], [614, 468], [714, 470], [703, 382], (659, 423)],
    #     'ID_7': [[702, 397], [702, 463], [797, 470], [797, 403], (749, 433)]
    # }

    roi_centers = [val[4] for val in ROI.values()]

    thickness = 1

    def __init__(self, model, tracking, csv_path):
        self.model = model
        self.csv_path = csv_path
        self.tracking = tracking

    def csv_logging(self, csv_path, data):
        logging_time = datetime.now().strftime(self.time_fmt)
        logging_time = logging_time.replace(" ", "_").replace(":", "_")

        csv_filename = f"logging_{logging_time}.csv"
        csv_file = os.path.join(csv_path, csv_filename)

        # Check if the file exists to determine whether to write the header
        file_exists = os.path.isfile(csv_file)

        fieldnames = ["S.No.", 
                      "Vehicle_ID",
                      "Occupied Station",
                      "Entry",
                      "Exit",
                      "Duration"]

        with open(csv_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write the data row
            for count, track_id in enumerate(data.keys()):
                count = count+1
                entry = data[track_id]["Entry"]
                occupied_station = data[track_id]["Occupied Station"]
                duration = data[track_id]["Duration"]
                if "Exit" in data[track_id]:
                    exit = data[track_id]["Exit"]
                else:
                    exit = ""

                enter_data = {"S.No.": count, 
                              "Vehicle_ID": track_id,
                              "Occupied Station": occupied_station,
                              "Entry": entry,
                              "Exit": exit,
                              "Duration":duration}

                writer.writerow(enter_data)

    def timer(self, entry_time, current_time):

        # time formatting
        start_time = datetime.strptime(entry_time, self.time_fmt)
        end_time = datetime.strptime(current_time, self.time_fmt)

        time_diff = end_time - start_time

        sec = time_diff.total_seconds()

        time = str(timedelta(seconds= sec))

        return time


    def occupancy_analysis(self, detected_points: list, center_coord: tuple, track_ids= None):    
        points_dist = []
        dist = 0
        track_id = None

        # measuring distance between the car and ROI
        for i, point in enumerate(detected_points):
            dist = math.dist(point, center_coord)
            points_dist.append(dist)
            
            dist = min(points_dist)
            min_dist_index = points_dist.index(dist)

#            min_center = detected_points[min_dist_index]
            dist = int(dist)

            if track_ids is not None:
                track_id = track_ids[min_dist_index]

        detected_coords = detected_points[min_dist_index]
        points_dist.clear()
        
        return dist, detected_coords, track_id
    
    def process_frame(self, frame, tracker):
        occupied_status = []

        detections, detection_centroids = self.detection(frame, tracker)

        tracker_ids = list(detections.tracker_id)

        for i, (key, val) in enumerate(self.ROI.items()):    
            x0, y0, x1, y1 = int(val[0][0]), int(val[0][1]), int(val[2][0]), int(val[2][1])
            pt1, pt2 = (x0, y0), (x1, y1)

            roi_centre = self.roi_centers[i]

            distance, _ , track_id = self.occupancy_analysis(detection_centroids, roi_centre, tracker_ids)

            if distance <25:
                occupied_status.append(1)

                color = self.red

                frame = self.process_detected_pbjects(frame, detections)

                self.session_log[track_id]["Occupied Station"] = key
                
                cv2.rectangle(frame, (x0, y0-20), (x0 + 40, y0), color, -1)
                cv2.putText(frame, key, (x0, y0-5), cv2.FONT_HERSHEY_DUPLEX, 0.4, self.white, 1)

                start_time = self.session_log[track_id]["Entry"]
                current_time = datetime.now().strftime(self.time_fmt)

                duration = self.timer(start_time, current_time)

                self.session_log[track_id]["Duration"] = duration

                cv2.putText(frame, duration, (x1-40, y0-10), cv2.FONT_HERSHEY_DUPLEX, 0.4, self.yellow, 1)
            
            else:
                occupied_status.append(0)
                color = self.green

                cv2.rectangle(frame, (x0, y0-20), (x0 + 40, y0), color, -1)
                cv2.putText(frame, key, (x0, y0-5), cv2.FONT_HERSHEY_DUPLEX, 0.4, self.white, 1)

                if len(self.occupied_station_status) > 0:
                    if self.occupied_station_status[0][i] != self.occupied_station_status[1][i]:

                        exit_timestamp = datetime.now().strftime(self.time_fmt)

                        if track_id in self.session_log:
                            self.session_log[track_id]["Exit"] = exit_timestamp


            cv2.rectangle(frame, pt1, pt2, color, self.thickness)

        self.occupied_station_status.append(occupied_status)
        
        return frame
    
    def process_detected_pbjects(self, frame, detections):
        
        detections_bboxes = detections.xyxy
        tracker_ids = detections.tracker_id
        # detected_classes = detections.data['class_name']

        for i, (bbox, track_id) in enumerate(zip(detections_bboxes, tracker_ids)):
            x0, y0, x1, y1 = map(int, bbox)
            pt1, pt2 = (x0, y0), (x1, y1)

            centre_coord = (int((x0+x1)/2), int((y0+y1)/2))

            distance, _, _ = self.occupancy_analysis(self.roi_centers, centre_coord)

            if distance < 25:
                
                entry_timestamp = datetime.now().strftime(self.time_fmt)
                
                track_label = f"#{track_id}"
                
                cv2.putText(frame, track_label, (x1-40, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.4, self.yellow, 1)
                cv2.rectangle(frame, pt1, pt2, self.yellow, 1)

                if track_id not in self.session_log:
                    self.session_log[track_id] = {"Entry": entry_timestamp}

        return frame
    
    def detection(self, frame, tracker):
        detection_centroids = []

        class_id_map = {name: idx for idx, name in self.model.names.items()}
        allowed_ids = [class_id_map[name] for name in self.tracking]

        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = [class_id in allowed_ids for class_id in detections.class_id]

        detections = detections[mask]
        detections = tracker.update_with_detections(detections)

        for detection in detections:
            x0, y0, x1, y1 = detection[0][0], detection[0][1], detection[0][2], detection[0][3]

            center_x, center_y = int((x0+x1)/2), int((y0+y1)/2)
            detection_centroids.append((center_x, center_y))
            
        return detections, detection_centroids

    def run_video(self, video_path):
        video_info = sv.VideoInfo.from_video_path(video_path)

        fps = video_info.fps

        tracker = sv.ByteTrack(minimum_matching_threshold=0.6, frame_rate=fps)

        run = True

        cap = cv2.VideoCapture(video_path)

        while run:
            ret, frame = cap.read()

            if ret:
                frame = self.process_frame(frame, tracker)
                cv2.imshow('feed', frame)

            k = cv2.waitKey(1)
            if k == ord('q') or not ret:
                run = False
                self.csv_logging(self.csv_path, self.session_log)
                   
        cap.release()
        cv2.destroyAllWindows()