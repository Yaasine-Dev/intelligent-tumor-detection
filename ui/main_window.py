import os
import tempfile
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QFont, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from model.cnn_model import CNNModel
from model.gradcam import GradCAM
from llm.llm_client import GroqClient
from utils.pdf_export import PDFExporter
from utils.preprocess import load_image, preprocess_for_model, preprocess_for_gradcam

IMG_DISPLAY = 300


# ── Worker thread ──────────────────────────────────────────────────────────────

class _AnalysisSignals(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)


class _AnalysisWorker(QRunnable):
    """Exécute CNN + Grad-CAM + LLM hors du thread UI."""

    def __init__(self, image_path: str, cnn: CNNModel, gradcam: GradCAM, llm: GroqClient):
        super().__init__()
        self.image_path = image_path
        self.cnn        = cnn
        self.gradcam    = gradcam
        self.llm        = llm
        self.signals    = _AnalysisSignals()

    @pyqtSlot()
    def run(self):
        try:
            # 1. Prédiction
            prediction = self.cnn.predict(self.image_path)

            # 2. Grad-CAM
            img_array     = load_image(self.image_path)
            gradcam_input = preprocess_for_gradcam(img_array)
            heatmap       = self.gradcam.generate(gradcam_input, class_index=None)
            _, overlay    = self.gradcam.overlay_heatmap(img_array, heatmap, alpha=0.4)

            # Sauvegarde overlay dans un fichier temporaire
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            overlay_bgr = overlay if overlay.shape[2] == 3 else cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(tmp.name, overlay_bgr)

            # 3. Rapport LLM
            llm_report = self.llm.generate_report(prediction)

            self.signals.finished.emit({
                "prediction":   prediction,
                "gradcam_path": tmp.name,
                "llm_report":   llm_report,
            })
        except Exception as exc:
            self.signals.error.emit(str(exc))


# ── Image display helper ───────────────────────────────────────────────────────

def _pixmap_from_path(path: str, size: int = IMG_DISPLAY) -> QPixmap:
    pix = QPixmap(path)
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _image_label(placeholder: str) -> QLabel:
    lbl = QLabel(placeholder)
    lbl.setFixedSize(IMG_DISPLAY, IMG_DISPLAY)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setObjectName("imageLabel")
    return lbl


# ── Main Window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Brain Tumor Detection System")
        self.setMinimumSize(1200, 720)
        self.setAcceptDrops(True)

        self._image_path: str | None = None
        self._gradcam_path: str | None = None
        self._prediction: dict | None = None
        self._llm_report: str = ""

        self._load_models()
        self._apply_styles()
        self._build_ui()

    # ── Init ───────────────────────────────────────────────────────────────────

    def _load_models(self):
        try:
            self._cnn     = CNNModel()
            self._gradcam = GradCAM(self._cnn.model)
            self._llm     = GroqClient()
        except Exception as exc:
            QMessageBox.critical(self, "Erreur de chargement", str(exc))
            raise

    def _apply_styles(self):
        qss_path = os.path.join(os.path.dirname(__file__), "styles.qss")
        if os.path.exists(qss_path):
            with open(qss_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

    # ── UI ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        main_layout = QHBoxLayout(root)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)

        main_layout.addWidget(self._build_left(),   stretch=1)
        main_layout.addWidget(self._build_center(), stretch=2)
        main_layout.addWidget(self._build_right(),  stretch=2)

        self.setCentralWidget(root)

    # ── Zone gauche ────────────────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("leftPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        title = QLabel("📂 Charger une IRM")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        self._drop_zone = QLabel("Glissez une image\nou cliquez sur le bouton")
        self._drop_zone.setObjectName("dropZone")
        self._drop_zone.setAlignment(Qt.AlignCenter)
        self._drop_zone.setFixedHeight(140)
        self._drop_zone.setWordWrap(True)
        layout.addWidget(self._drop_zone)

        btn_load = QPushButton("📁  Charger IRM")
        btn_load.setObjectName("btnPrimary")
        btn_load.clicked.connect(self._on_load_image)
        layout.addWidget(btn_load)

        self._file_label = QLabel("Aucun fichier sélectionné")
        self._file_label.setObjectName("fileLabel")
        self._file_label.setWordWrap(True)
        self._file_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._file_label)

        layout.addStretch()
        return panel

    # ── Zone centre ────────────────────────────────────────────────────────────

    def _build_center(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("centerPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        title = QLabel("🧠 Visualisation")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        images_row = QHBoxLayout()
        images_row.setSpacing(16)

        orig_box = QVBoxLayout()
        orig_box.addWidget(QLabel("Image originale"), alignment=Qt.AlignCenter)
        self._lbl_original = _image_label("Aucune image")
        orig_box.addWidget(self._lbl_original)
        images_row.addLayout(orig_box)

        gcam_box = QVBoxLayout()
        gcam_box.addWidget(QLabel("Grad-CAM"), alignment=Qt.AlignCenter)
        self._lbl_gradcam = _image_label("—")
        gcam_box.addWidget(self._lbl_gradcam)
        images_row.addLayout(gcam_box)

        layout.addLayout(images_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # mode indéterminé
        self._progress.setVisible(False)
        self._progress.setObjectName("progressBar")
        layout.addWidget(self._progress)

        self._btn_analyze = QPushButton("🔍  Analyser")
        self._btn_analyze.setObjectName("btnAnalyze")
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.clicked.connect(self._on_analyze)
        layout.addWidget(self._btn_analyze)

        layout.addStretch()
        return panel

    # ── Zone droite ────────────────────────────────────────────────────────────

    def _build_right(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        title = QLabel("📋 Résultats")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        # Résultat prédiction
        result_box = QGroupBox("Résultat de la prédiction")
        result_box.setObjectName("resultBox")
        result_layout = QVBoxLayout(result_box)

        self._lbl_class = QLabel("—")
        self._lbl_class.setObjectName("predClass")
        self._lbl_class.setAlignment(Qt.AlignCenter)

        self._lbl_confidence = QLabel("")
        self._lbl_confidence.setObjectName("predConfidence")
        self._lbl_confidence.setAlignment(Qt.AlignCenter)

        result_layout.addWidget(self._lbl_class)
        result_layout.addWidget(self._lbl_confidence)
        layout.addWidget(result_box)

        # Rapport médical
        report_box = QGroupBox("Rapport Médical")
        report_box.setObjectName("reportBox")
        report_layout = QVBoxLayout(report_box)

        self._txt_report = QTextEdit()
        self._txt_report.setReadOnly(True)
        self._txt_report.setPlaceholderText("Le rapport médical apparaîtra ici après l'analyse...")
        self._txt_report.setObjectName("reportText")
        report_layout.addWidget(self._txt_report)
        layout.addWidget(report_box, stretch=1)

        # Export PDF
        self._btn_pdf = QPushButton("📄  Exporter PDF")
        self._btn_pdf.setObjectName("btnPdf")
        self._btn_pdf.setEnabled(False)
        self._btn_pdf.clicked.connect(self._on_export_pdf)
        layout.addWidget(self._btn_pdf)

        return panel

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir une IRM",
            "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp)"
        )
        if path:
            self._set_image(path)

    def _on_analyze(self):
        if not self._image_path:
            return
        self._set_analyzing(True)

        worker = _AnalysisWorker(self._image_path, self._cnn, self._gradcam, self._llm)
        worker.signals.finished.connect(self._on_analysis_done)
        worker.signals.error.connect(self._on_analysis_error)
        QThreadPool.globalInstance().start(worker)

    def _on_analysis_done(self, result: dict):
        self._set_analyzing(False)

        self._prediction   = result["prediction"]
        self._gradcam_path = result["gradcam_path"]
        self._llm_report   = result["llm_report"]

        # Grad-CAM
        self._lbl_gradcam.setPixmap(_pixmap_from_path(self._gradcam_path))

        # Prédiction
        cls  = self._prediction["class"]
        conf = self._prediction["confidence"]
        self._lbl_class.setText(cls)
        self._lbl_confidence.setText(f"Confiance : {conf:.1f} %")

        # Rapport
        self._txt_report.setMarkdown(self._llm_report)

        self._btn_pdf.setEnabled(True)

    def _on_analysis_error(self, msg: str):
        self._set_analyzing(False)
        QMessageBox.critical(self, "Erreur d'analyse", msg)

    def _on_export_pdf(self):
        if not self._prediction:
            return

        default_name = f"rapport_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le rapport PDF",
            default_name, "PDF (*.pdf)"
        )
        if not path:
            return

        try:
            PDFExporter().generate_report(
                data={
                    "patient_name": "—",
                    "date":         datetime.now().strftime("%d/%m/%Y"),
                    "image_path":   self._image_path,
                    "prediction":   self._prediction,
                    "gradcam_path": self._gradcam_path,
                    "llm_report":   self._llm_report,
                },
                output_path=path,
            )
            QMessageBox.information(self, "Export réussi", f"Rapport enregistré :\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur export PDF", str(exc))

    # ── Drag & Drop ────────────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            self._set_image(urls[0].toLocalFile())

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _set_image(self, path: str):
        self._image_path = path
        self._file_label.setText(os.path.basename(path))
        self._lbl_original.setPixmap(_pixmap_from_path(path))
        self._lbl_gradcam.setText("—")
        self._lbl_class.setText("—")
        self._lbl_confidence.setText("")
        self._txt_report.clear()
        self._btn_analyze.setEnabled(True)
        self._btn_pdf.setEnabled(False)
        self._prediction   = None
        self._gradcam_path = None

    def _set_analyzing(self, active: bool):
        self._btn_analyze.setEnabled(not active)
        self._progress.setVisible(active)
        if active:
            self._lbl_class.setText("Analyse en cours…")
            self._lbl_confidence.setText("")
            self._txt_report.setPlaceholderText("Génération du rapport…")
