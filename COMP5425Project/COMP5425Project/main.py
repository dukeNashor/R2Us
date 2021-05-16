from SBIR import *
from PyQt5 import QtCore, QtGui, QtWidgets
from assets.sbirmainwindow import *

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    ui = Ui_SBIRMainWindow()
    ui.setupUi(main_window)
    uim = UIManager(ui)

    main_window.show()
    sys.exit(app.exec_())
