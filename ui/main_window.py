from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intelligent Brain Tumor Detection System")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        self.label = QLabel("Welcome to Intelligent Brain Tumor Detection System")
        layout.addWidget(self.label)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
