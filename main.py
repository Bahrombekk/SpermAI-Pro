import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QFrame, QStackedWidget, QProgressBar, QSplitter,
                           QGraphicsDropShadowEffect, QDialog, QFormLayout, 
                           QLineEdit, QTextEdit, QDialogButtonBox, QMessageBox)
from PyQt6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QIcon, QPixmap, QImage, QPalette, QColor, QFont, QScreen
import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
from pathlib import Path

class ModernButton(QPushButton):
    def __init__(self, text, icon_path=None, gradient=False):
        super().__init__(text)
        self.setMinimumHeight(45)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        if icon_path and os.path.exists(icon_path):
            self.setIcon(QIcon(icon_path))
            self.setIconSize(QSize(24, 24))
        
        style = """
            QPushButton {
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                margin: 2px;
            }
        """
        
        if gradient:
            style += """
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                              stop:0 #4F46E5, stop:1 #7C3AED);
                    color: white;
                    border: none;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                              stop:0 #4338CA, stop:1 #6D28D9);
                }
            """
        else:
            style += """
                QPushButton {
                    background-color: #1F2937;
                    color: #9CA3AF;
                    border: 1px solid #374151;
                }
                QPushButton:hover {
                    background-color: #374151;
                    color: white;
                    border: 1px solid #4B5563;
                }
            """
        
        self.setStyleSheet(style)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

class StatsCard(QFrame):
    def __init__(self, title, icon_path=None, gradient_colors=None):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #4F46E5, stop:1 #7C3AED);
                border-radius: 8px;
                padding: 10px;
            }
        """)
        header_layout = QHBoxLayout(header)
        
        if icon_path and os.path.exists(icon_path):
            icon = QLabel()
            icon.setPixmap(QPixmap(icon_path).scaled(24, 24))
            header_layout.addWidget(icon)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        self.value_label = QLabel("0%")
        self.value_label.setStyleSheet("color: white; font-size: 32px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.change_label = QLabel("")
        self.change_label.setStyleSheet("color: #34D399; font-size: 14px;")
        self.change_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(header)
        layout.addWidget(self.value_label)
        layout.addWidget(self.change_label)
        
        self.setStyleSheet("""
            StatsCard {
                background-color: #1F2937;
                border-radius: 12px;
                padding: 0px;
                min-height: 160px;
            }
        """)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
    
    def update_values(self, value, change=None):
        self.value_label.setText(f"{value}%")
        if change:
            color = "#34D399" if change.startswith("+") else "#EF4444"
            self.change_label.setStyleSheet(f"color: {color}; font-size: 14px;")
            self.change_label.setText(change)

class PatientInfoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bemor ma'lumotlari")
        self.setMinimumWidth(400)
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.name_input = QLineEdit()
        self.birth_date_input = QLineEdit()
        self.birth_date_input.setPlaceholderText("KK.OO.YYYY")
        self.id_input = QLineEdit()
        self.conclusion_input = QTextEdit()
        self.doctor_input = QLineEdit()

        layout.addRow("Bemor F.I.SH:", self.name_input)
        layout.addRow("Tug'ilgan sanasi:", self.birth_date_input)
        layout.addRow("ID raqami:", self.id_input)
        layout.addRow("XULOSA:", self.conclusion_input)
        layout.addRow("Shifokor:", self.doctor_input)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)

    def get_data(self):
        return {
            'name': self.name_input.text(),
            'birth_date': self.birth_date_input.text(),
            'id': self.id_input.text(),
            'conclusion': self.conclusion_input.toPlainText(),
            'doctor': self.doctor_input.text()
        }

class SpermAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
        model_path = "best(1).pt"  # Your model path here
        try:
            self.model = YOLO(model_path)
            self.status_label.setText("Model muvaffaqiyatli yuklandi")
        except Exception as e:
            self.status_label.setText(f"Model yuklanishida xatolik: {str(e)}")
            self.model = None

    def init_ui(self):
        self.setWindowTitle("SpermAI Analysis Pro")
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                          stop:0 #111827, stop:1 #1F2937);
            }
        """)
        
        screen = QApplication.primaryScreen().size()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))
        self.center_window()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("""
            #sidebar {
                background-color: rgba(31, 41, 55, 0.7);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setSpacing(15)
        
        logo_layout = QHBoxLayout()
        logo_label = QLabel("🔬")
        logo_label.setStyleSheet("font-size: 32px;")
        title_label = QLabel("SpermAI Pro")
        title_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        logo_layout.addWidget(logo_label)
        logo_layout.addWidget(title_label)
        logo_layout.addStretch()
        sidebar_layout.addLayout(logo_layout)
        
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #374151;")
        sidebar_layout.addWidget(separator)
        
        sidebar_buttons = [
            ("Dashboard", "icons/dashboard.png", True),
            ("Yangi Tahlil", "icons/microscope.png", False),
            ("Statistika", "icons/chart.png", False),
            ("Hisobotlar", "icons/document.png", False)
        ]
        
        for text, icon, is_active in sidebar_buttons:
            btn = ModernButton(text, icon, gradient=is_active)
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()
        splitter.addWidget(sidebar)
        
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)
        
        stats_layout = QHBoxLayout()
        self.trik_card = StatsCard("Trik Spermalar", "icons/live.png")
        self.ulik_card = StatsCard("O'lik Spermalar", "icons/dead.png")
        self.yetilmagan_card = StatsCard("Yetilmagan", "icons/immature.png")
        
        stats_layout.addWidget(self.trik_card)
        stats_layout.addWidget(self.ulik_card)
        stats_layout.addWidget(self.yetilmagan_card)
        content_layout.addLayout(stats_layout)
        
        analysis_layout = QHBoxLayout()
        
        upload_frame = QFrame()
        upload_frame.setObjectName("uploadFrame")
        upload_frame.setStyleSheet("""
            #uploadFrame {
                background-color: rgba(31, 41, 55, 0.7);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        upload_layout = QVBoxLayout(upload_frame)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #4B5563;
                border-radius: 15px;
                background-color: rgba(17, 24, 39, 0.5);
            }
        """)
        
        upload_btn = ModernButton("Rasm tanlash", gradient=True)
        upload_btn.clicked.connect(self.load_image)
        
        upload_layout.addWidget(self.image_label)
        upload_layout.addWidget(upload_btn)
        
        results_frame = QFrame()
        results_frame.setObjectName("resultsFrame")
        results_frame.setStyleSheet("""
            #resultsFrame {
                background-color: rgba(31, 41, 55, 0.7);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        results_layout = QVBoxLayout(results_frame)
        
        status_panel = QFrame()
        status_panel.setStyleSheet("""
            QFrame {
                background-color: rgba(59, 130, 246, 0.1);
                border-radius: 10px;
                padding: 15px;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }
        """)
        status_layout = QVBoxLayout(status_panel)
        
        self.status_label = QLabel("Tahlil natijasi")
        self.status_label.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #374151;
                border-radius: 5px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #4F46E5;
                border-radius: 5px;
            }
        """)
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        results_layout.addWidget(status_panel)
        
        self.results_list = QWidget()
        results_list_layout = QVBoxLayout(self.results_list)
        self.results_list.setVisible(False)
        results_layout.addWidget(self.results_list)
        results_layout.addStretch()
        
        actions_layout = QHBoxLayout()
        report_btn = ModernButton("Hisobot yaratish")
        report_btn.clicked.connect(self.create_report)
        save_btn = ModernButton("Natijalarni saqlash", gradient=True)
        save_btn.clicked.connect(self.save_results)
        
        actions_layout.addWidget(report_btn)
        actions_layout.addWidget(save_btn)
        results_layout.addLayout(actions_layout)
        
        analysis_layout.addWidget(upload_frame)
        analysis_layout.addWidget(results_frame)
        content_layout.addLayout(analysis_layout)
        
        splitter.addWidget(content)
# Continue from where main_layout left off
        main_layout.addWidget(splitter)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)

    def center_window(self):
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        window_size = self.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        self.move(x, y)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Rasmni tanlang",
            "",
            "Rasm fayllari (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_name:
            self.current_image_path = file_name
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.analyze_image()

    def analyze_image(self):
        if not hasattr(self, 'current_image_path') or not self.model:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Progress simulation
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(50)
        
        # Perform analysis with YOLO model
        try:
            results = self.model(self.current_image_path)[0]
            self.process_results(results)
        except Exception as e:
            QMessageBox.critical(self, "Xatolik", f"Tahlil jarayonida xatolik: {str(e)}")
            self.progress_bar.setVisible(False)

    def update_progress(self):
        current_value = self.progress_bar.value()
        if current_value >= 100:
            self.progress_timer.stop()
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setValue(current_value + 1)

    def process_results(self, results):
        if not results.boxes:
            self.status_label.setText("Hech qanday sperma topilmadi")
            return

        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        total_count = len(classes)
        live_count = np.sum(classes == 0)
        dead_count = np.sum(classes == 1)
        immature_count = np.sum(classes == 2)

        # Update stats cards
        self.trik_card.update_values(
            round((live_count / total_count) * 100),
            f"+{live_count} dona"
        )
        self.ulik_card.update_values(
            round((dead_count / total_count) * 100),
            f"+{dead_count} dona"
        )
        self.yetilmagan_card.update_values(
            round((immature_count / total_count) * 100),
            f"+{immature_count} dona"
        )

        self.status_label.setText("Tahlil muvaffaqiyatli yakunlandi")
        self.results_list.setVisible(True)

    def create_report(self):
        dialog = PatientInfoDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            patient_data = dialog.get_data()
            self.save_report(patient_data)

    def save_report(self, patient_data):
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "Ogohlantirish", "Avval rasm yuklang!")
            return

        try:
            # Create reports directory if it doesn't exist
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # Create report filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_report_path = reports_dir / f"report_{timestamp}.txt"
            html_report_path = reports_dir / f"report_{timestamp}.html"

            # Get current values
            trik_value = self.trik_card.value_label.text().replace('%', '')
            ulik_value = self.ulik_card.value_label.text().replace('%', '')
            yetilmagan_value = self.yetilmagan_card.value_label.text().replace('%', '')
            
            # Get counts from change labels
            trik_count = self.trik_card.change_label.text().replace('+', '').replace(' dona', '')
            ulik_count = self.ulik_card.change_label.text().replace('+', '').replace(' dona', '')
            yetilmagan_count = self.yetilmagan_card.change_label.text().replace('+', '').replace(' dona', '')
            
            total_count = int(trik_count) + int(ulik_count) + int(yetilmagan_count)

            # Save TXT report
            with open(txt_report_path, "w", encoding="utf-8") as f:
                f.write("=== SPERM TAHLILI NATIJASI ===\n\n")
                f.write(f"Sana: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n")
                f.write(f"Bemor: {patient_data['name']}\n")
                f.write(f"Tug'ilgan sana: {patient_data['birth_date']}\n")
                f.write(f"ID: {patient_data['id']}\n\n")
                f.write("--- NATIJALAR ---\n")
                f.write(f"Trik spermalar: {trik_value}%\n")
                f.write(f"O'lik spermalar: {ulik_value}%\n")
                f.write(f"Yetilmagan spermalar: {yetilmagan_value}%\n\n")
                f.write("XULOSA:\n")
                f.write(f"{patient_data['conclusion']}\n\n")
                f.write(f"Shifokor: {patient_data['doctor']}")

            # Read HTML template
            with open("index.html", "r", encoding="utf-8") as f:
                html_template = f.read()

            # Replace values in HTML
            current_time = datetime.now().strftime('%d.%m.%Y %H:%M')
            html_content = html_template.replace('Teshaboyev Teshavoy Teshavoyevich', patient_data['name'])
            html_content = html_content.replace('01.01.1990', patient_data['birth_date'])
            html_content = html_content.replace('19.10.2024', current_time.split()[0])
            html_content = html_content.replace('10:30', current_time.split()[1])
            html_content = html_content.replace('SP-2024/123', patient_data['id'])
            
            # Update table values
            html_content = html_content.replace('<td>500</td>\n                <td>50%</td>', 
                                             f'<td>{trik_count}</td>\n                <td>{trik_value}%</td>')
            html_content = html_content.replace('<td>0</td>\n                <td>0%</td>', 
                                             f'<td>{ulik_count}</td>\n                <td>{ulik_value}%</td>')
            html_content = html_content.replace('<td>500</td>\n                <td>50%</td>', 
                                             f'<td>{yetilmagan_count}</td>\n                <td>{yetilmagan_value}%</td>')
            html_content = html_content.replace('<td>1000</td>', f'<td>{total_count}</td>')

            # Update conclusion and doctor
            html_content = html_content.replace('Trik spermatozoidlar miqdori me\'yordan biroz past.', 
                                             patient_data['conclusion'])
            html_content = html_content.replace('__________________</td>\n                    <td width="50"></td>', 
                                             f'{patient_data["doctor"]}</td>\n                    <td width="50"></td>')

            # Save HTML report
            with open(html_report_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            # Copy the HTML report to index.html to display current results
            with open("index.html", "w", encoding="utf-8") as f:
                f.write(html_content)

            QMessageBox.information(
                self,
                "Muvaffaqiyat",
                f"Hisobot saqlandi:\nTXT: {txt_report_path}\nHTML: {html_report_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Xatolik",
                f"Hisobotni saqlashda xatolik: {str(e)}"
            )

    def save_results(self):
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "Ogohlantirish", "Avval rasm yuklang!")
            return

        try:
            # Create results directory if it doesn't exist
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)

            # Save analyzed image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = results_dir / f"analysis_{timestamp}.jpg"
            
            # Save results data
            data_path = results_dir / f"data_{timestamp}.txt"
            
            with open(data_path, "w", encoding="utf-8") as f:
                f.write("=== TAHLIL NATIJALARI ===\n")
                f.write(f"Sana: {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n")
                f.write(f"Trik spermalar: {self.trik_card.value_label.text()}\n")
                f.write(f"O'lik spermalar: {self.ulik_card.value_label.text()}\n")
                f.write(f"Yetilmagan spermalar: {self.yetilmagan_card.value_label.text()}\n")

            # Copy the analyzed image
            import shutil
            shutil.copy2(self.current_image_path, image_path)

            QMessageBox.information(
                self,
                "Muvaffaqiyat",
                f"Natijalar saqlandi:\nRasm: {image_path}\nMa'lumotlar: {data_path}"
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Xatolik",
                f"Natijalarni saqlashda xatolik: {str(e)}"
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SpermAnalysisApp()
    window.show()
    sys.exit(app.exec())