import os
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "tumor_model.keras")
QSS_PATH   = os.path.join(os.path.dirname(__file__), "ui", "styles.qss")


def _load_stylesheet(app: QApplication) -> None:
    if os.path.exists(QSS_PATH):
        with open(QSS_PATH, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())


def _check_model(app: QApplication) -> bool:
    if not os.path.exists(MODEL_PATH):
        QMessageBox.warning(
            None,
            "Modèle manquant",
            f"Le fichier du modèle est introuvable :\n{MODEL_PATH}\n\n"
            "Placez 'tumor_model.keras' dans le dossier model/ et relancez l'application.",
        )
        return False
    return True


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Intelligent Brain Tumor Detection")
    app.setOrganizationName("Medical AI Lab")

    _load_stylesheet(app)

    if not _check_model(app):
        sys.exit(1)

    from ui.main_window import MainWindow   # import tardif : modèle vérifié avant TF
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
