from PyQt5.QtWidgets import QMainWindow, QApplication
from mainwindow import App
import sys


def main(args):
    app = QApplication(args)
    window = App()
    window.show()
    app.exec()


if __name__ == "__main__":
    main(sys.argv)