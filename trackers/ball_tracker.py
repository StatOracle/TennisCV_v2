from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate().bfill()
        return [{1: x} for x in df_ball_positions.to_numpy().tolist()]

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2
        df['mid_y_rolling_mean'] = df['mid_y'].rolling(window=5, min_periods=1).mean()
        df['delta_y'] = df['mid_y_rolling_mean'].diff()
        
        min_change_frames = 25
        for i in range(1, len(df) - int(min_change_frames * 1.2)):
            negative_change = df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[i+1] < 0
            positive_change = df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[i+1] > 0

            if negative_change or positive_change:
                change_count = sum(
                    (df['delta_y'].iloc[i] > 0 and df['delta_y'].iloc[j] < 0) or
                    (df['delta_y'].iloc[i] < 0 and df['delta_y'].iloc[j] > 0)
                    for j in range(i+1, i+int(min_change_frames * 1.2) + 1)
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
        return {1: results.boxes[0].xyxy.tolist()[0]} if results.boxes else {1: []}

    def draw_bboxes(self, video_frames, player_detections):
        output_frames = []
        for frame, ball_dict in zip(video_frames, player_detections):
            for track_id, bbox in ball_dict.items():
                if bbox:  # Ensure bbox is not empty
                    x1, y1, x2, y2 = bbox
                    cv2.putText(frame, f"Ball ID: {track_id}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_frames.append(frame)
        return output_frames
