import pickle
import cv2
from ultralytics import YOLO

class PlayerTracker:
    def __init__(self, model_path):
        self.yolo = YOLO(model_path)

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        player_detections = []
        
        for frame in frames:
            player_dict = self.track(frame)  # 🔥 FIXED: Calls `track`, not `detect_frames`
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections  # 🔥 FIXED: Returns detections

    def track(self, frame):
        results = self.yolo.track(frame, persist=True)  # 🔥 FIXED: `self.yolo` instead of `self.model`
        if results[0].boxes is None:  # Handle cases with no detections
            return {}

        id_name_dict = self.yolo.names  # 🔥 FIXED: Get class names
        player_dict = {}

        for box in results[0].boxes:
            if box.id is None:
                continue  # Skip boxes without a tracking ID
            
            track_id = int(box.id.tolist()[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == "person":
                player_dict[track_id] = (x1, y1, x2, y2)

        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player {track_id}", 
                            (int(x1), int(y1) - 10),  # 🔥 FIXED: Corrected text position
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, 
                              (int(x1), int(y1)), 
                              (int(x2), int(y2)), 
                              (0, 255, 0), 2)

            output_video_frames.append(frame)  # 🔥 FIXED: Append only once per frame
        
        return output_video_frames
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
    
        filtered_detections = []

        for frame_detections in player_detections:
            if not frame_detections:
                filtered_detections.append({})
                continue

        # Sort by bounding box area (width * height), descending
            sorted_players = sorted(
                frame_detections.items(),
                key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
                reverse=True
        )

        # Take top 2
            filtered = dict(sorted_players[:2])
            filtered_detections.append(filtered)

        return filtered_detections
