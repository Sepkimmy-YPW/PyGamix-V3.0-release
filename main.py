from logic import *

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app_icon = QIcon("./setting/icon.png")
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    app.setFont(QFont("Bahnschrift", 12))
    win = Mainwindow()
    win.show()
    sys.exit(app.exec_())