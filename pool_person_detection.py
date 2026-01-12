"""
YOLO ile Havuz Etrafındaki İnsan Tespiti
Bu script, YOLOv8 kullanarak havuz etrafındaki insanları tespit eder.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


from collections import defaultdict
import time

class PoolPersonDetector:
    """Havuz etrafındaki insanları ve potansiyel tehlikeleri tespit eden sınıf"""
    
    def __init__(self, model_path='yolov8n-pose.pt', conf_threshold=0.25):
        """
        Args:
            model_path: YOLO model dosyası yolu (varsayılan: yolov8n-pose.pt)
            conf_threshold: Güven eşiği (0-1 arası)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Tracking history
        self.track_history = defaultdict(lambda: {
            'positions': [], # List of (x, y) tuples
            'last_seen': 0,
            'danger_score': 0,
            'status': 'Normal' # Normal, Warning, Drowning
        })
        
        # Parameters for heuristics
        self.fps = 30 # Will be updated from video
        self.stationary_threshold_seconds = 5.0 # Seconds to trigger stationary warning
        self.drowning_threshold_seconds = 10.0 # Seconds to trigger drowning alert
        self.movement_threshold = 20 # Pixels (minimum movement to be considered "moving")
        
    def detect_and_track(self, image):
        """
        Görüntüdeki insanları takip eder ve pozlarını analiz eder
        """
        # YOLO ile takip yap (persist=True önemli)
        results = self.model.track(image, persist=True, conf=self.conf_threshold, verbose=False)
        
        # Use original image (not .plot()) to avoid duplicate labels
        annotated_image = image.copy()
        
        current_time = time.time()
        
        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()

            # Check if keypoints exist (pose model vs detection model)
            has_keypoints = results[0].keypoints is not None
            keypoints_data = results[0].keypoints.data.cpu() if has_keypoints else None

            for i, (box, track_id, cls_id, conf) in enumerate(zip(boxes, track_ids, class_ids, confs)):
                x, y, w, h = box
                center = (float(x), float(y))
                class_name = self.model.names[cls_id]

                # Get keypoints for this person if available
                kpts = keypoints_data[i] if has_keypoints else None

                # Update history
                history = self.track_history[track_id]
                history['last_seen'] = current_time
                history['positions'].append(center)

                # Keep only recent history (last 10 seconds approx)
                max_history = int(self.fps * 15)
                if len(history['positions']) > max_history:
                    history['positions'] = history['positions'][-max_history:]

                # Draw keypoints (skeleton) first
                if kpts is not None:
                    self._draw_keypoints(annotated_image, kpts)

                # Analyze Status
                status, color = self._analyze_status(track_id, kpts)

                # Draw Status on Image
                self._draw_status(annotated_image, box, status, color, track_id, class_name, conf)
                
        return results, annotated_image
    
    def _draw_keypoints(self, image, keypoints):
        """
        Keypoint'leri ve iskelet baglantilarini cizer
        COCO 17 keypoint formati
        """
        # COCO skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # bacaklar
            [6, 12], [7, 13],  # govde-bacak
            [6, 7],  # omuzlar
            [6, 8], [7, 9],  # kollar ust
            [8, 10], [9, 11],  # kollar alt
            [2, 3], [1, 2], [1, 3],  # yuz
            [2, 4], [3, 5],  # kulaklar
            [4, 6], [5, 7]  # kulak-omuz
        ]

        # Keypoint renkleri (sag-sol farkli)
        kpt_color = (0, 255, 255)  # Sari
        left_color = (255, 128, 0)  # Turuncu (sol taraf)
        right_color = (51, 153, 255)  # Mavi (sag taraf)

        # Keypoint'leri ciz
        for idx, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if conf > 0.5:
                color = kpt_color
                if idx in [1, 3, 5, 7, 9, 11, 13, 15]:  # Sol taraf
                    color = left_color
                elif idx in [2, 4, 6, 8, 10, 12, 14, 16]:  # Sag taraf
                    color = right_color
                cv2.circle(image, (int(x), int(y)), 3, color, -1)
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 0), 1)

        # Iskelet cizgilerini ciz
        for connection in skeleton:
            idx1, idx2 = connection[0] - 1, connection[1] - 1  # 0-indexed
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
                if kpt1[2] > 0.5 and kpt2[2] > 0.5:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

    def _analyze_status(self, track_id, keypoints):
        """
        Analyze person status based on movement and pose
        """
        history = self.track_history[track_id]
        positions = history['positions']
        
        # Check if we have enough data
        min_frames = int(self.fps * 2) # Check last 2 seconds
        if len(positions) < min_frames:
            return "Analyzing...", (255, 255, 0)
            
        # 1. Movement Analysis (Stationary check)
        # Compare current position with position 2 seconds ago
        recent_pos = positions[-min_frames:]
        start_pos = recent_pos[0]
        end_pos = recent_pos[-1]
        
        movement = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        is_stationary = movement < self.movement_threshold
        
        # 2. Pose Analysis (Head visibility check) - only if keypoints available
        head_visible = True  # Default to True if no keypoints (can't detect drowning)
        if keypoints is not None:
            # COCO Keypoints: 0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
            head_indices = [0, 1, 2, 3, 4]
            head_visible = False
            for idx in head_indices:
                if keypoints.shape[0] > idx and keypoints[idx][2] > 0.5:
                    head_visible = True
                    break
        
        # Update Danger Score
        if is_stationary:
            history['danger_score'] += 1
        else:
            # Decrease danger score if moving (recovery)
            history['danger_score'] = max(0, history['danger_score'] - 2)
            
        score = history['danger_score']
        
        # Determine Status
        if score > (self.drowning_threshold_seconds * self.fps):
            if not head_visible:
                return "DROWNING ALERT!", (0, 0, 255) # Red
            else:
                return "STATIONARY (Danger)", (0, 165, 255) # Orange
        elif score > (self.stationary_threshold_seconds * self.fps):
            return "Stationary", (0, 255, 255) # Yellow
        else:
            return "Active", (0, 255, 0) # Green

    def _draw_status(self, image, box, status, color, track_id, class_name="", conf=0.0):
        x, y, w, h = box

        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)

        # Renk paleti (class'a göre)
        if class_name == 'child':
            box_color = (255, 147, 0)  # Turuncu
            label_bg = (255, 147, 0)
        elif class_name == 'adult':
            box_color = (0, 200, 255)  # Cyan
            label_bg = (0, 200, 255)
        else:
            box_color = (147, 20, 255)  # Mor
            label_bg = (147, 20, 255)

        # Yarı saydam bbox dolgusu
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

        # Bbox çerçevesi
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1, cv2.LINE_AA)

        # Label
        label = f"{class_name} {conf:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Pill şeklinde label arka planı
        padding_x = 6
        padding_y = 4
        label_x1 = x1
        label_y1 = y1 - th - padding_y * 2
        label_x2 = x1 + tw + padding_x * 2
        label_y2 = y1

        # Rounded rectangle için
        radius = 5
        overlay2 = image.copy()
        cv2.rectangle(overlay2, (label_x1 + radius, label_y1), (label_x2 - radius, label_y2), label_bg, -1)
        cv2.rectangle(overlay2, (label_x1, label_y1 + radius), (label_x2, label_y2 - radius), label_bg, -1)
        cv2.circle(overlay2, (label_x1 + radius, label_y1 + radius), radius, label_bg, -1)
        cv2.circle(overlay2, (label_x2 - radius, label_y1 + radius), radius, label_bg, -1)
        cv2.circle(overlay2, (label_x1 + radius, label_y2 - radius), radius, label_bg, -1)
        cv2.circle(overlay2, (label_x2 - radius, label_y2 - radius), radius, label_bg, -1)
        cv2.addWeighted(overlay2, 0.85, image, 0.15, 0, image)

        # Label text (gölgeli)
        text_x = label_x1 + padding_x
        text_y = label_y2 - padding_y
        cv2.putText(image, label, (text_x+1, text_y+1), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    
    def process_video(self, video_path, output_path=None, show_preview=True):
        """Video dosyasındaki insanları takip eder"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Hata: Video dosyası açılamadı: {video_path}")
            return
        
        # Update FPS
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Çıkış video yazıcısı
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        frame_count = 0
        
        print(f"Video işleniyor: {video_path}")
        print(f"Çözünürlük: {width}x{height}, FPS: {self.fps}")
        print("Çıkmak için 'q' tuşuna basın.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and Track
            results, annotated_frame = self.detect_and_track(frame)
            
            # Info overlay
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Save and Show
            if out:
                out.write(annotated_frame)
            
            if show_preview:
                cv2.imshow('Havuz Guvenlik Takibi', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Kullanıcı tarafından durduruldu")
                    break
            
            frame_count += 1
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"\nİşlem tamamlandı! Toplam frame: {frame_count}")

    def process_image(self, image_path, output_path=None, show_preview=True):
        """Görüntü işleme - Takip geçmişi olmadan sadece anlık analiz"""
        # Not: Tek resimde hareket analizi yapılamaz, sadece poz analizi yapılır
        image = cv2.imread(image_path)
        if image is None: 
            print("Hata: Resim açılamadı")
            return
            
        results, annotated_image = self.detect_and_track(image)
        
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Kaydedildi: {output_path}")
            
        if show_preview:
            cv2.imshow('Sonuc', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def process_webcam(self, camera_index=0):
        """Webcam üzerinden canlı takip"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Hata: Webcam açılamadı")
            return
            
        self.fps = 30 # Webcam varsayılan
        print("Webcam başlatıldı...")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            results, annotated_frame = self.detect_and_track(frame)
            
            cv2.imshow('Canlı Havuz Takibi', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLO ile Havuz Etrafındaki İnsan Tespiti')
    parser.add_argument('--input', '-i', type=str, help='Giriş dosyası (görüntü veya video)')
    parser.add_argument('--output', '-o', type=str, help='Çıkış dosyası yolu')
    parser.add_argument('--model', '-m', type=str, default='yolov8n-pose.pt', 
                       help='YOLO model dosyası (varsayılan: yolov8n-pose.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25, 
                       help='Güven eşiği (varsayılan: 0.25)')
    parser.add_argument('--webcam', '-w', action='store_true', 
                       help='Webcam kullan (giriş dosyası yerine)')
    parser.add_argument('--no-preview', action='store_true', 
                       help='Önizlemeyi gösterme')
    
    args = parser.parse_args()
    
    # Detector oluştur
    detector = PoolPersonDetector(model_path=args.model, conf_threshold=args.conf)
    
    # Webcam modu
    if args.webcam:
        detector.process_webcam()
    # Giriş dosyası var
    elif args.input:
        input_path = Path(args.input)
        
        if not input_path.exists():
            print(f"Hata: Dosya bulunamadı: {args.input}")
            return
        
        # Dosya tipine göre işle
        if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            detector.process_image(args.input, args.output, show_preview=not args.no_preview)
        elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            detector.process_video(args.input, args.output, show_preview=not args.no_preview)
        else:
            print(f"Hata: Desteklenmeyen dosya formatı: {input_path.suffix}")
    else:
        print("Hata: Giriş dosyası veya --webcam parametresi gerekli.")
        print("Kullanım: python pool_person_detection.py --input <dosya> [--output <çıkış>]")
        print("         python pool_person_detection.py --webcam")


if __name__ == "__main__":
    main()

