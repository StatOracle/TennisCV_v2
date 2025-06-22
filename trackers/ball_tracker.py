from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        raw = [x.get(1, []) for x in ball_positions]
        valid = [pos for pos in raw if pos]

        if not valid:
            print("⚠️ Warning: No valid ball positions to interpolate.")
            return ball_positions

        df = pd.DataFrame(valid, columns=['x1', 'y1', 'x2', 'y2']).interpolate().bfill()

        output = []
        j = 0
        for pos in raw:
            if pos:
                output.append({1: df.iloc[j].tolist()})
                j += 1
            else:
                output.append({1: []})
        return output

    def get_ball_shot_frames(self, ball_positions):
        cleaned = [x.get(1, []) for x in ball_positions if x.get(1)]

        if not cleaned:
            print("⚠️ Error: No valid ball positions found for shot detection.")
            return []

        df = pd.DataFrame(cleaned, columns=['x1', 'y1', 'x2', 'y2'])
        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1).mean()
        df['delta_y'] = df['mid_y_rolling_mean'].diff()

        min_change_frames = 25
        for i in range(1, len(df) - int(min_change_frames * 1.2)):
            dy_now = df['delta_y'].iloc[i]
            dy_next = df['delta_y'].iloc[i + 1]
            if (dy_now > 0 and dy_next < 0) or (dy_now < 0 and dy_next > 0):
                change_count = sum(
                    (dy_now > 0 and df['delta_y'].iloc[j] < 0) or
                    (dy_now < 0 and df['delta_y'].iloc[j] > 0)
                    for j in range(i + 1, i + int(min_change_frames * 1.2) + 1)
                    if j < len(df)
                )
                if change_count > min_change_frames - 1:
                    df.loc[i, 'ball_hit'] = 1

        return df[df['ball_hit'] == 1].index.tolist()

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path:
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        ball_detections = [self.detect_frame(frame) for frame in frames]

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return {1: []}
        
        for box in results.boxes:
            cls_id = int(box.cls.tolist()[0])
            cls_name = self.model.names[cls_id].lower()
            if "ball" in cls_name or "sports ball" in cls_name:
                return {1: box.xyxy[0].tolist()}
        
        return {1: []}

    def draw_bboxes(self, video_frames, ball_detections):
        output_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_frames.append(frame)
        return output_frames
