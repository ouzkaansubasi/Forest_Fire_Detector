# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\oukaa\Desktop\fire\personel.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lblisim = QtWidgets.QLabel(self.centralwidget)
        self.lblisim.setGeometry(QtCore.QRect(30, 60, 55, 16))
        self.lblisim.setObjectName("lblisim")
        self.lblsoyisim = QtWidgets.QLabel(self.centralwidget)
        self.lblsoyisim.setGeometry(QtCore.QRect(30, 90, 55, 16))
        self.lblsoyisim.setObjectName("lblsoyisim")
        self.lblemail = QtWidgets.QLabel(self.centralwidget)
        self.lblemail.setGeometry(QtCore.QRect(30, 130, 55, 16))
        self.lblemail.setObjectName("lblemail")
        self.txtisim = QtWidgets.QLineEdit(self.centralwidget)
        self.txtisim.setGeometry(QtCore.QRect(100, 60, 151, 21))
        self.txtisim.setObjectName("txtisim")
        self.txtsoyisim = QtWidgets.QLineEdit(self.centralwidget)
        self.txtsoyisim.setGeometry(QtCore.QRect(100, 90, 151, 21))
        self.txtsoyisim.setObjectName("txtsoyisim")
        self.txtemail = QtWidgets.QLineEdit(self.centralwidget)
        self.txtemail.setGeometry(QtCore.QRect(110, 130, 151, 21))
        self.txtemail.setObjectName("txtemail")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(30, 230, 741, 311))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.btnkaydet = QtWidgets.QPushButton(self.centralwidget)
        self.btnkaydet.setGeometry(QtCore.QRect(40, 170, 81, 31))
        self.btnkaydet.setObjectName("btnkaydet")
        self.btnsil = QtWidgets.QPushButton(self.centralwidget)
        self.btnsil.setGeometry(QtCore.QRect(160, 170, 81, 31))
        self.btnsil.setObjectName("btnsil")
        self.btnguncelle = QtWidgets.QPushButton(self.centralwidget)
        self.btnguncelle.setGeometry(QtCore.QRect(270, 170, 81, 31))
        self.btnguncelle.setObjectName("btnguncelle")
        self.btnlistele = QtWidgets.QPushButton(self.centralwidget)
        self.btnlistele.setGeometry(QtCore.QRect(380, 170, 81, 31))
        self.btnlistele.setObjectName("btnlistele")
        self.lblid = QtWidgets.QLabel(self.centralwidget)
        self.lblid.setGeometry(QtCore.QRect(30, 20, 55, 16))
        self.lblid.setObjectName("lblid")
        self.txtid = QtWidgets.QLineEdit(self.centralwidget)
        self.txtid.setGeometry(QtCore.QRect(90, 20, 141, 22))
        self.txtid.setObjectName("txtid")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.menuPersonel = QtWidgets.QMenu(self.menubar)
        self.menuPersonel.setObjectName("menuPersonel")
        self.menuAna_Sayfa = QtWidgets.QMenu(self.menubar)
        self.menuAna_Sayfa.setObjectName("menuAna_Sayfa")
        MainWindow.setMenuBar(self.menubar)
        self.menubar.addAction(self.menuAna_Sayfa.menuAction())
        self.menubar.addAction(self.menuPersonel.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.lblisim.setText(_translate("MainWindow", "İsim :"))
        self.lblsoyisim.setText(_translate("MainWindow", "Soyisim :"))
        self.lblemail.setText(_translate("MainWindow", "E-Mail :"))
        self.btnkaydet.setText(_translate("MainWindow", "Kaydet"))
        self.btnsil.setText(_translate("MainWindow", "Sil"))
        self.btnguncelle.setText(_translate("MainWindow", "Güncelle"))
        self.btnlistele.setText(_translate("MainWindow", "Listele"))
        self.lblid.setText(_translate("MainWindow", "ID :"))
        self.menuPersonel.setTitle(_translate("MainWindow", "Personel"))
        self.menuAna_Sayfa.setTitle(_translate("MainWindow", "Ana Sayfa"))