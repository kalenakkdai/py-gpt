from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QPushButton, QHBoxLayout, QWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

class ResultDialog(QDialog):
    def __init__(self, parent=None):
        super(ResultDialog, self).__init__(parent)
        self.setWindowTitle("Image Analysis Results")
        self.setGeometry(100, 100, 1200, 800)

        self.layout = QVBoxLayout(self)

        self.result_list = QListWidget(self)
        self.layout.addWidget(self.result_list)

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close)
        self.layout.addWidget(self.close_button)

    def add_result(self, result: str, image_path: str):
        """Add a new result to the list with text wrapping and image display"""
        item = QListWidgetItem(self.result_list)
        
        # Create a widget to hold the image and text
        widget = QWidget()
        widget_layout = QHBoxLayout(widget)
        
        # Image display
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        image_label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        widget_layout.addWidget(image_label)
        
        # Text display
        text_label = QLabel(result)
        text_label.setWordWrap(True)
        text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        widget_layout.addWidget(text_label)
        
        # Set the widget as the item widget
        item.setSizeHint(widget.sizeHint())
        self.result_list.addItem(item)
        self.result_list.setItemWidget(item, widget) 