import cv2
import numpy as np
import sys
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QWidget, QFrame, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt

class ImageCaptioningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Captioning with YOLO & BLIP")
        self.setGeometry(100, 100, 1200, 800)
        
        # Load models
        self.yolo_model = YOLO('yolo12n.pt')
        
        # Load BLIP captioning model (smaller version for CPU)
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                           stop:0 #f6f8fa, stop:1 #e9ecef);
            }
        """)
        
        # Header
        header_label = QLabel("Image Captioning with YOLO & BLIP")
        header_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        header_label.setStyleSheet("""
            color: #1a5276; 
            margin: 20px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            border: 2px solid #3498db;
        """)
        main_layout.addWidget(header_label)
        
        # Content layout
        content_layout = QHBoxLayout()
        
        # Left panel - Image display
        left_panel = QFrame()
        left_panel.setFrameShape(QFrame.StyledPanel)
        left_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.85); 
                border-radius: 15px;
                border: 2px solid #3498db;
                padding: 10px;
            }
        """)
        left_layout = QVBoxLayout(left_panel)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 500)
        self.image_label.setStyleSheet("""
            border: 3px dashed #3498DB; 
            border-radius: 10px;
            background-color: #f8f9fa;
            color: #7f8c8d;
            font-size: 16px;
            font-weight: bold;
        """)
        
        upload_btn = QPushButton("Upload Image")
        upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #2980B9;
                border: 2px solid #1a5276;
            }
        """)
        upload_btn.clicked.connect(self.upload_image)
        
        process_btn = QPushButton("Generate Caption")
        process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ECC71;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #27AE60;
                border: 2px solid #1e8449;
            }
        """)
        process_btn.clicked.connect(self.process_image)
        
        save_btn = QPushButton("Save Results")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
                border: 2px solid #6c3483;
            }
        """)
        save_btn.clicked.connect(self.save_results)
        
        left_layout.addWidget(self.image_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(upload_btn)
        btn_layout.addWidget(process_btn)
        btn_layout.addWidget(save_btn)
        left_layout.addLayout(btn_layout)
        
        # Right panel - Results
        right_panel = QFrame()
        right_panel.setFrameShape(QFrame.StyledPanel)
        right_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.85); 
                border-radius: 15px;
                border: 2px solid #e74c3c;
                padding: 10px;
            }
        """)
        right_layout = QVBoxLayout(right_panel)
        
        results_label = QLabel("Detection & Caption Results")
        results_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        results_label.setAlignment(Qt.AlignCenter)
        results_label.setStyleSheet("""
            color: #c0392b; 
            margin: 10px;
            padding: 5px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 8px;
        """)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 2px solid #BDC3C7;
                border-radius: 10px;
                padding: 15px;
                font-size: 14px;
                font-family: 'Segoe UI';
                color: #34495e;
            }
        """)
        
        right_layout.addWidget(results_label)
        right_layout.addWidget(self.results_text)
        
        # Add panels to content layout
        content_layout.addWidget(left_panel, 3)
        content_layout.addWidget(right_panel, 2)
        
        main_layout.addLayout(content_layout)
        
        self.original_image = None
        self.processed_image = None
        
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image)
            self.results_text.clear()
            self.results_text.append("<span style='color:#3498db; font-weight:bold; font-size:15px;'>Image loaded successfully!</span>")
            self.results_text.append("<span style='color:#7f8c8d;'>Click 'Generate Caption' to process.</span>")
            
    def process_image(self):
        if self.original_image is None:
            self.results_text.clear()
            self.results_text.append("<span style='color:#e74c3c; font-weight:bold; font-size:15px;'>Please upload an image first!</span>")
            return
        
        self.results_text.clear()
        self.results_text.append("<span style='color:#f39c12; font-weight:bold; font-size:15px;'>Processing... (this may take a moment on CPU)</span>")
        QApplication.processEvents()  # Update UI
            
        # YOLO detection
        results = self.yolo_model(self.original_image)
        
        # Process YOLO results and draw boxes
        display_img = self.original_image.copy()
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates and class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls]
                
                # Draw box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_img, f"{class_name}: {conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                detected_objects.append(class_name)
        
        # BLIP image captioning
        image = Image.fromarray(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        
        # Generate caption
        inputs = self.caption_processor(image, return_tensors="pt")
        
        # Add context based on detected objects for better captions
        if detected_objects:
            detected_set = set(detected_objects)
            context = f"an image with {', '.join(detected_set)}"
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        else:
            # Generate without context if no objects detected
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        
        # Save processed image for later saving
        self.processed_image = display_img
        
        # Display results
        self.display_image(display_img)
        self.results_text.clear()
        
        # Show detected objects
        self.results_text.append("<span style='color:#2c3e50; font-weight:bold; font-size:16px;'>Detected Objects:</span>")
        if detected_objects:
            for obj in set(detected_objects):
                self.results_text.append(f"<span style='color:#2980b9; font-size:14px;'>• {obj}</span>")
        else:
            self.results_text.append("<span style='color:#7f8c8d; font-style:italic;'>No objects detected by YOLO model</span>")
        
        self.results_text.append("<br><span style='color:#2c3e50; font-weight:bold; font-size:16px;'>Generated Caption:</span>")
        self.results_text.append(f"<span style='color:#16a085; font-size:15px;'>{caption}</span>")
        self.results_text.append("<br><span style='color:#7f8c8d; font-style:italic;'>Click 'Save Results' to save the image with detections and caption.</span>")
            
    def display_image(self, img):
        h, w, c = img.shape
        bytes_per_line = 3 * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_label.setAlignment(Qt.AlignCenter)
    
    def save_results(self):
        if self.processed_image is None:
            self.results_text.append("<span style='color:#e74c3c; font-weight:bold;'>Please process an image first!</span>")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            # Save the image with detections
            cv2.imwrite(file_path, self.processed_image)
            
            # Save caption text alongside the image
            text_file_path = file_path.rsplit('.', 1)[0] + "_caption.txt"
            with open(text_file_path, 'w') as f:
                f.write(self.results_text.toPlainText())
                
            self.results_text.append("<br><span style='color:#27ae60; font-weight:bold;'>Results saved successfully!</span>")
            self.results_text.append(f"<span style='color:#7f8c8d;'>Image: {file_path}</span>")
            self.results_text.append(f"<span style='color:#7f8c8d;'>Caption: {text_file_path}</span>")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageCaptioningApp()
    window.show()
    sys.exit(app.exec_())