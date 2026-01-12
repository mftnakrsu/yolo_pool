"""
Video bölme ve işleme scripti
"""
import cv2
import os
from pool_person_detection import PoolPersonDetector

def split_video(input_path, output_dir, segment_duration=6):
    """Videoyu segment_duration saniyelik parçalara böler"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_per_segment = fps * segment_duration
    segment_count = 0
    frame_count = 0

    out = None
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_per_segment == 0:
            if out:
                out.release()
            segment_count += 1
            output_path = os.path.join(output_dir, f"{base_name}_part{segment_count}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Segment {segment_count} oluşturuluyor: {output_path}")

        out.write(frame)
        frame_count += 1

    if out:
        out.release()
    cap.release()
    print(f"Toplam {segment_count} segment oluşturuldu")
    return segment_count

def process_and_split(input_path, model_path, output_dir, segment_duration=6):
    """Videoyu işle ve böl"""
    os.makedirs(output_dir, exist_ok=True)

    detector = PoolPersonDetector(model_path=model_path, conf_threshold=0.25)

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_per_segment = fps * segment_duration
    segment_count = 0
    frame_count = 0

    out = None
    base_name = "output"

    print(f"Video işleniyor: {input_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_per_segment == 0:
            if out:
                out.release()
            segment_count += 1
            output_path = os.path.join(output_dir, f"{base_name}_part{segment_count}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output segment {segment_count}: {output_path}")

        # YOLO ile işle
        results, annotated_frame = detector.detect_and_track(frame)
        out.write(annotated_frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  Frame {frame_count} işlendi...")

    if out:
        out.release()
    cap.release()
    print(f"Toplam {segment_count} output segment oluşturuldu")

if __name__ == "__main__":
    # Klasörler
    os.makedirs("website/videos/original", exist_ok=True)
    os.makedirs("website/videos/output", exist_ok=True)

    # 1. Orijinal videoyu böl
    print("\n=== Orijinal video bölünüyor ===")
    split_video("demo_havuz.mov", "website/videos/original", segment_duration=6)

    # 2. İşlenmiş videoyu oluştur ve böl
    print("\n=== Video işlenip bölünüyor ===")
    process_and_split("demo_havuz.mov", "yolov8m.pt", "website/videos/output", segment_duration=6)

    print("\nTamamlandı!")
