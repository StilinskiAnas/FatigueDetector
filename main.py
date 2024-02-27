import sys
import cv2
from PyQt6.QtWidgets import QApplication, QStatusBar, QGroupBox, QWidget, QVBoxLayout, QCheckBox, QLabel, QSlider, \
    QGridLayout, QHBoxLayout
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from face_action import FaceAction
import numpy as np
import csv
import os


class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.face_action = FaceAction()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

        # Add a timer to trigger the saving of person_data every 5 seconds
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(self.save_person_data)
        self.save_interval = 5000  # milliseconds (5 seconds)
        self.save_timer.start(self.save_interval)

        # List to store the last detected person_data within the interval
        self.last_detected_person_data = []

    def initUI(self):
        # Disposition
        disposition = QVBoxLayout()

        # Cases à cocher
        self.cb_grayscale = QCheckBox('Niveau de gris')
        self.cb_blur = QCheckBox('Flou gaussien')
        self.cb_brightness = QCheckBox('Luminosité')
        self.cb_contrast = QCheckBox('Contraste')
        self.cb_hist_eq = QCheckBox('Égalisation d\'histogramme')
        self.cb_detect_fatigue = QCheckBox('Détection de fatigue')
        self.cb_canny = QCheckBox('Détection de bord Canny')
        self.cb_dilation = QCheckBox('Dilatation')
        self.cb_erosion = QCheckBox('Érosion')

        # Curseurs
        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_contrast = QSlider(Qt.Orientation.Horizontal)
        self.slider_kernel_size = QSlider(Qt.Orientation.Horizontal)

        self.slider_brightness.setMinimum(0)
        self.slider_brightness.setMaximum(255)
        self.slider_brightness.setValue(127)
        self.slider_contrast.setMinimum(0)
        self.slider_contrast.setMaximum(255)
        self.slider_contrast.setValue(0)
        self.slider_kernel_size.setMinimum(3)
        self.slider_kernel_size.setMaximum(9)
        self.slider_kernel_size.setValue(5)
        self.slider_kernel_size.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_kernel_size.setTickInterval(2)

        # Étiquettes
        self.label_image = QLabel(self)
        self.label_fatigue = QLabel(self)
        self.label_kernel_size = QLabel('Taille du noyau : 3')
        self.label_brightness = QLabel('Luminosité : 127')
        self.label_contrast = QLabel('Contraste : 0')

        # Tableau
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(["Personne", "œil gauche", "œil droite", "EAR", "MAR"])
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Connecter les signaux aux emplacements
        self.cb_dilation.stateChanged.connect(self.update_dilation_erosion)
        self.cb_erosion.stateChanged.connect(self.update_dilation_erosion)
        self.slider_kernel_size.valueChanged.connect(self.update_kernel_size)
        self.slider_brightness.valueChanged.connect(self.update_brightness_contrast)
        self.slider_contrast.valueChanged.connect(self.update_brightness_contrast)
        self.cb_hist_eq.stateChanged.connect(self.update_hist_eq)
        self.cb_detect_fatigue.stateChanged.connect(self.update_detect_fatigue)

        # Attributs
        self.brightness = 0
        self.contrast = 1
        self.hist_eq = False
        self.canny_enabled = False
        self.detect_fatigue_enabled = False
        self.dilation_enabled = False
        self.erosion_enabled = False
        self.kernel_size = 3

        # Créer une disposition horizontale pour les cases à cocher
        hbox = QHBoxLayout()
        hbox.addWidget(self.cb_grayscale)
        hbox.addWidget(self.cb_blur)
        hbox.addWidget(self.cb_brightness)
        hbox.addWidget(self.cb_contrast)
        hbox.addWidget(self.cb_hist_eq)
        hbox.addWidget(self.cb_canny)
        hbox.addWidget(self.cb_detect_fatigue)
        hbox.addWidget(self.cb_dilation)
        hbox.addWidget(self.cb_erosion)

        # Créer une disposition en grille pour les curseurs et les étiquettes
        grid = QGridLayout()
        grid.addWidget(self.label_brightness, 0, 0)
        grid.addWidget(self.slider_brightness, 0, 1)
        grid.addWidget(self.label_contrast, 1, 0)
        grid.addWidget(self.slider_contrast, 1, 1)
        grid.addWidget(self.label_kernel_size, 2, 0)
        grid.addWidget(self.slider_kernel_size, 2, 1)

        # Créer une disposition verticale pour l'image et le tableau
        vbox = QVBoxLayout()
        vbox.addWidget(self.label_image)
        vbox.addWidget(self.tableWidget)

        # Créer des groupes pour chaque disposition
        gb_checkboxes = QGroupBox("Options")
        gb_checkboxes.setLayout(hbox)

        # Créer une barre d'état pour les alertes
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Aucune fatigue détectée")
        self.status_bar.setStyleSheet("background-color: green")

        gb_sliders = QGroupBox("Paramètres")
        gb_sliders.setLayout(grid)

        gb_image = QGroupBox("Image")
        gb_image.setLayout(vbox)

        # Ajouter les groupes et la barre d'état à la disposition principale
        disposition.addWidget(gb_checkboxes)
        disposition.addWidget(gb_sliders)
        disposition.addWidget(self.status_bar)
        disposition.addWidget(gb_image)

        # Définir la disposition principale
        self.setLayout(disposition)

    def update_frame(self):
        # Lire une image de la séquence vidéo
        ret, frame = self.cap.read()
        if not ret:
            return

            # Appliquer les transformations en fonction des cases à cocher
        frame = self.apply_grayscale(frame)
        frame = self.apply_blur(frame)
        frame = self.apply_brightness(frame)
        frame = self.apply_contrast(frame)
        frame = self.apply_hist_eq(frame)
        frame = self.apply_canny_edge_detection(frame)
        frame = self.apply_dilation_erosion(frame)
        frame = self.apply_detect_fatigue(frame)

        # Convertir en QImage et afficher
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_image.setPixmap(QPixmap.fromImage(q_img))

    def apply_grayscale(self, frame):
        # Convertir l'image en niveaux de gris si la case à cocher est activée
        if self.cb_grayscale.isChecked():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def apply_blur(self, frame):
        # Appliquer un flou gaussien à l'image si la case à cocher est activée
        if self.cb_blur.isChecked():
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return frame

    def apply_brightness(self, frame):
        # Ajuster la luminosité de l'image en utilisant la valeur du curseur
        if self.cb_brightness.isChecked():
            frame = cv2.add(frame, self.brightness)
            frame = np.clip(frame, 0, 255)
        return frame

    def apply_contrast(self, frame):
        # Ajuster le contraste de l'image en utilisant la valeur du curseur
        if self.cb_contrast.isChecked():
            frame = cv2.multiply(frame, self.contrast)
            frame = np.clip(frame, 0, 255)
        return frame

    def apply_hist_eq(self, frame):
        # Appliquer une égalisation d'histogramme à chaque canal et les fusionner
        if self.cb_hist_eq.isChecked():
            channels = cv2.split(frame)
            equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
            frame = cv2.merge(equalized_channels)
        return frame

    def apply_detect_fatigue(self, frame):
        # Faire passer l'image à travers l'analyse d'action faciale et afficher les résultats
        if self.cb_detect_fatigue.isChecked():
            results = self.face_action.run_frame(frame)
            table_data = []
            fatigue_count = 0  # Compter le nombre de personnes fatiguées
            for i, (ear, mar, leftEye, rightEye, mouth, leftEAR, rightEAR) in enumerate(results, start=1):
                person_data = [
                    f"Personne:{i} ",
                    f"{leftEAR:.2f} ",
                    f"{rightEAR:.2f} ",
                    f"{ear:.2f} ",
                    f"{mar:.2f} "
                ]
                for point in rightEye:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                for point in leftEye:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                for point in mouth:
                    cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
                cv2.putText(frame, f"œil gauche : {leftEAR:.2f}", (leftEye[0][0] - 190, leftEye[0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"œil droite : {rightEAR:.2f}", (leftEye[0][0] + 40, leftEye[0][1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                # Vérifier si la personne est fatiguée en fonction des valeurs de ear et mar
                if ear < 0.25 or mar > 0.7:
                    fatigue_count += 1  # Incrémenter le compteur de fatigue
                table_data.append(person_data)
            # Update the last detected person_data within the interval
            self.last_detected_person_data = table_data

            # Mettre à jour le tableau avec les nouvelles données
            self.update_table(table_data)

            # Mettre à jour le message de la barre d'état et la couleur
            if fatigue_count == 0:
                self.status_bar.showMessage("Aucune fatigue détectée")
                self.status_bar.setStyleSheet("background-color: green")
            elif fatigue_count == 1:
                self.status_bar.showMessage("Fatigue détectée chez 1 personne")
                self.status_bar.setStyleSheet("background-color: yellow")
            else:
                self.status_bar.showMessage(f"Fatigue détectée chez {fatigue_count} personnes")
                self.status_bar.setStyleSheet("background-color: red")
        return frame

    def closeEvent(self, event):
        # Libérer l'objet de capture vidéo et fermer toutes les fenêtres
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

    def update_brightness_contrast(self):
        # Mettre à jour les valeurs de luminosité et de contraste en fonction des valeurs des curseurs
        self.brightness = self.slider_brightness.value() - 127
        self.contrast = 1 + self.slider_contrast.value() / 127

    def update_hist_eq(self):
        # Mettre à jour le drapeau d'égalisation d'histogramme en fonction de l'état de la case à cocher
        self.hist_eq = self.cb_hist_eq.isChecked()

    def update_detect_fatigue(self):
        # Mettre à jour le drapeau de détection de fatigue en fonction de l'état de la case à cocher
        self.detect_fatigue_enabled = self.cb_detect_fatigue.isChecked()

    def apply_canny_edge_detection(self, frame):
        # S'assurer que l'image est au format d'entier non signé sur 8 bits
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Appliquer la détection de bord Canny à l'image si la case à cocher est activée
        if self.cb_canny.isChecked():
            frame = cv2.Canny(frame, 50, 150)
        return frame

    def apply_dilation_erosion(self, frame):
        # Appliquer la dilatation et l'érosion à l'image en fonction des cases à cocher et de la valeur du curseur
        if self.cb_dilation.isChecked():
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            frame = cv2.dilate(frame, kernel, iterations=1)

        if self.cb_erosion.isChecked():
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            frame = cv2.erode(frame, kernel, iterations=1)

        return frame

    def update_dilation_erosion(self):
        # Mettre à jour les drapeaux de dilatation et d'érosion en fonction de l'état des cases à cocher
        self.dilation_enabled = self.cb_dilation.isChecked()
        self.erosion_enabled = self.cb_erosion.isChecked()

    def update_kernel_size(self):
        # Mettre à jour la taille du noyau en fonction de la valeur du curseur
        self.kernel_size = self.slider_kernel_size.value()
        self.label_kernel_size.setText(f'Taille du noyau : {self.kernel_size}')

    def update_table(self, data):
        if (len(data) == 0):
            data = [[
                f"{0}",
                f"{0}",
                f"{0}",
                f"{0}",
                f"{0}"
            ]]

        self.tableWidget.setRowCount(len(data))

        self.tableWidget.setColumnCount(len(data[0]))

        for i, row in enumerate(data):
            for j, item in enumerate(row):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(item)))

    def save_person_data(self):
        # Save the last detected person_data to a CSV file
        filename = "person_data.csv"
        write_header = not os.path.exists(filename)

        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            if write_header:
                # Write header if the file didn't exist before
                csv_writer.writerow(["Person", "Left Ear", "Right Ear", "EAR", "MAR"])

            if self.last_detected_person_data:
                # Write the last detected person_data
                csv_writer.writerows(self.last_detected_person_data)

        # Clear the last_detected_person_data for the next interval
        self.last_detected_person_data = []


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()
    ex.show()
    sys.exit(app.exec())
