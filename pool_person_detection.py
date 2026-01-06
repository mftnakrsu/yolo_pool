"""
YOLO ile Havuz Etrafındaki İnsan Tespiti
Bu script, YOLOv8 kullanarak havuz etrafındaki insanları tespit eder.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import argparse


class PoolPersonDetector:
    """Havuz etrafındaki insanları tespit eden sınıf"""
    
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.25):
        """
        Args:
            model_path: YOLO model dosyası yolu (varsayılan: yolov8n.pt)
            conf_threshold: Güven eşiği (0-1 arası)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # COCO veri setinde 'person' sınıfı 0 numaralı sınıftır
        self.person_class_id = 0
        
    def detect_persons(self, image):
        """
        Görüntüdeki insanları tespit eder
        
        Args:
            image: OpenCV görüntü (numpy array)
            
        Returns:
            results: Tespit sonuçları
            annotated_image: Üzerine çizilmiş görüntü
        """
        # YOLO ile tespit yap
        results = self.model(image, conf=self.conf_threshold, classes=[self.person_class_id])
        
        # Sonuçları görüntüye çiz
        annotated_image = results[0].plot()
        
        return results, annotated_image
    
    def process_video(self, video_path, output_path=None, show_preview=True):
        """
        Video dosyasındaki insanları tespit eder
        
        Args:
            video_path: Giriş video dosyası yolu
            output_path: Çıkış video dosyası yolu (None ise kaydetmez)
            show_preview: Ekranda göster (True/False)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Hata: Video dosyası açılamadı: {video_path}")
            return
        
        # Video özelliklerini al
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Çıkış video yazıcısı
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_persons = 0
        
        print(f"Video işleniyor: {video_path}")
        print(f"Çözünürlük: {width}x{height}, FPS: {fps}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # İnsan tespiti yap
            results, annotated_frame = self.detect_persons(frame)
            
            # Tespit edilen insan sayısını al
            detections = results[0].boxes
            person_count = len(detections)
            total_persons += person_count
            
            # Bilgileri ekrana yazdır
            info_text = f"Frame: {frame_count} | Tespit Edilen İnsan: {person_count}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Çıkış videosuna yaz
            if out:
                out.write(annotated_frame)
            
            # Ekranda göster
            if show_preview:
                cv2.imshow('Havuz İnsan Tespiti', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Kullanıcı tarafından durduruldu")
                    break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"İşlenen frame: {frame_count}, Ortalama insan sayısı: {total_persons/frame_count:.2f}")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nİşlem tamamlandı!")
        print(f"Toplam frame: {frame_count}")
        print(f"Ortalama tespit edilen insan sayısı: {total_persons/frame_count:.2f}")
        if output_path:
            print(f"Çıkış video kaydedildi: {output_path}")
    
    def process_image(self, image_path, output_path=None, show_preview=True):
        """
        Görüntü dosyasındaki insanları tespit eder
        
        Args:
            image_path: Giriş görüntü dosyası yolu
            output_path: Çıkış görüntü dosyası yolu (None ise kaydetmez)
            show_preview: Ekranda göster (True/False)
        """
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Hata: Görüntü dosyası açılamadı: {image_path}")
            return
        
        # İnsan tespiti yap
        results, annotated_image = self.detect_persons(image)
        
        # Tespit edilen insan sayısını al
        detections = results[0].boxes
        person_count = len(detections)
        
        # Bilgileri ekrana yazdır
        info_text = f"Tespit Edilen İnsan: {person_count}"
        cv2.putText(annotated_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Çıkış görüntüsünü kaydet
        if output_path:
            cv2.imwrite(output_path, annotated_image)
            print(f"Sonuç kaydedildi: {output_path}")
        
        # Ekranda göster
        if show_preview:
            cv2.imshow('Havuz İnsan Tespiti', annotated_image)
            print(f"Tespit edilen insan sayısı: {person_count}")
            print("Çıkmak için bir tuşa basın...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return person_count
    
    def process_webcam(self, camera_index=0):
        """
        Webcam'den canlı görüntü alarak insan tespiti yapar
        
        Args:
            camera_index: Kamera indeksi (varsayılan: 0)
        """
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Hata: Kamera açılamadı: {camera_index}")
            return
        
        print("Webcam'den canlı tespit başlatıldı. Çıkmak için 'q' tuşuna basın.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # İnsan tespiti yap
            results, annotated_frame = self.detect_persons(frame)
            
            # Tespit edilen insan sayısını al
            detections = results[0].boxes
            person_count = len(detections)
            
            # Bilgileri ekrana yazdır
            info_text = f"Tespit Edilen İnsan: {person_count}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Havuz İnsan Tespiti - Canlı', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Canlı tespit durduruldu.")


def main():
    parser = argparse.ArgumentParser(description='YOLO ile Havuz Etrafındaki İnsan Tespiti')
    parser.add_argument('--input', '-i', type=str, help='Giriş dosyası (görüntü veya video)')
    parser.add_argument('--output', '-o', type=str, help='Çıkış dosyası yolu')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt', 
                       help='YOLO model dosyası (varsayılan: yolov8n.pt)')
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

