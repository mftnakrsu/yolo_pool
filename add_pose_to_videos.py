"""
Mevcut videolara YOLOv8 Pose Estimation ekler
outputs_custom klasoründeki child detection videolarına pose overlay ekler
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


class PoseEstimator:
    """YOLOv8 Pose Estimation ile video işleme"""

    def __init__(self, model_path='yolov8n-pose.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # COCO skeleton connections
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # bacaklar
            [6, 12], [7, 13],  # govde-bacak
            [6, 7],  # omuzlar
            [6, 8], [7, 9],  # kollar ust
            [8, 10], [9, 11],  # kollar alt
            [2, 3], [1, 2], [1, 3],  # yuz
            [2, 4], [3, 5],  # kulaklar
            [4, 6], [5, 7]  # kulak-omuz
        ]

        # Renk paleti
        self.kpt_color = (0, 255, 255)  # Sari
        self.left_color = (255, 128, 0)  # Turuncu (sol taraf)
        self.right_color = (51, 153, 255)  # Mavi (sag taraf)
        self.skeleton_color = (0, 255, 0)  # Yesil

    def draw_keypoints(self, image, keypoints):
        """Keypoint'leri ve iskelet baglantilarini cizer"""
        # Keypoint'leri ciz
        for idx, kpt in enumerate(keypoints):
            x, y, conf = kpt
            if conf > 0.5:
                color = self.kpt_color
                if idx in [1, 3, 5, 7, 9, 11, 13, 15]:  # Sol taraf
                    color = self.left_color
                elif idx in [2, 4, 6, 8, 10, 12, 14, 16]:  # Sag taraf
                    color = self.right_color
                cv2.circle(image, (int(x), int(y)), 4, color, -1)
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 0), 1)

        # Iskelet cizgilerini ciz
        for connection in self.skeleton:
            idx1, idx2 = connection[0] - 1, connection[1] - 1  # 0-indexed
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                kpt1, kpt2 = keypoints[idx1], keypoints[idx2]
                if kpt1[2] > 0.5 and kpt2[2] > 0.5:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(image, pt1, pt2, self.skeleton_color, 2, cv2.LINE_AA)

    def process_frame(self, frame):
        """Tek bir frame'e pose estimation uygular"""
        # YOLOv8 pose detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        annotated = frame.copy()

        if results[0].keypoints is not None:
            keypoints_data = results[0].keypoints.data.cpu().numpy()

            for person_kpts in keypoints_data:
                self.draw_keypoints(annotated, person_kpts)

        return annotated

    def process_video(self, input_path, output_path=None, show_preview=False):
        """Video dosyasına pose estimation ekler"""
        cap = cv2.VideoCapture(str(input_path))

        if not cap.isOpened():
            print(f"Hata: Video acilamadi: {input_path}")
            return False

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if output_path is None:
            input_p = Path(input_path)
            output_path = input_p.parent / f"{input_p.stem}_pose{input_p.suffix}"

        # Video writer
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"\nIslem: {Path(input_path).name}")
        print(f"Cozunurluk: {width}x{height}, FPS: {fps}, Toplam frame: {total_frames}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pose estimation uygula
            annotated = self.process_frame(frame)

            # Kaydet
            out.write(annotated)

            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Ilerleme: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')

            if show_preview:
                cv2.imshow('Pose Estimation', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"\n  Tamamlandi! Cikti: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description='Videolara YOLOv8 Pose Estimation ekle')
    parser.add_argument('--input', '-i', type=str, help='Tek video dosyasi')
    parser.add_argument('--folder', '-f', type=str, help='Video klasoru')
    parser.add_argument('--output', '-o', type=str, help='Cikti dosyasi/klasoru')
    parser.add_argument('--model', '-m', type=str, default='yolov8n-pose.pt',
                       help='YOLO pose model (varsayilan: yolov8n-pose.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Guven esigi (varsayilan: 0.25)')
    parser.add_argument('--preview', action='store_true', help='Onizleme goster')

    args = parser.parse_args()

    estimator = PoseEstimator(model_path=args.model, conf_threshold=args.conf)

    if args.folder:
        folder = Path(args.folder)
        videos = list(folder.glob('*.mp4')) + list(folder.glob('*.avi')) + list(folder.glob('*.mov'))

        print(f"Bulunan video sayisi: {len(videos)}")

        for video_path in videos:
            # _pose ile biten dosyalari atla
            if '_pose' in video_path.stem:
                print(f"Atlaniyor (zaten pose): {video_path.name}")
                continue
            estimator.process_video(video_path, show_preview=args.preview)

    elif args.input:
        estimator.process_video(args.input, args.output, show_preview=args.preview)
    else:
        print("Hata: --input veya --folder parametresi gerekli")
        print("Ornek: python add_pose_to_videos.py --folder 'Risk Kademelendirmesi/outputs_custom'")


if __name__ == "__main__":
    main()
