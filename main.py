from ultralytics import YOLO
from video_analytics import VideoAnalytics


def main(model_path, tracking, csv_path):

    model = YOLO(model_path, task='detect')
    
    analytics = VideoAnalytics(model, tracking, csv_path)
    
    analytics.run_video(video_path)
    
model_path = r'path\to\model'
video_path = r"path\to\video"
csv_path = r'log'

tracking = ['car']

if __name__ == "__main__":
    main(model_path=model_path, tracking=tracking, csv_path=csv_path)
