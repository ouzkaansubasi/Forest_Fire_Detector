from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem

from Ui_personel import Ui_MainWindow

import sys
import os
import sqlite3 as sql

os.system("python connection.py")
os.system("python createTable.py")

global id, isim, soyisim, email

class Window(QtWidgets.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.formYukle()
        self.ui.btnkaydet.clicked.connect(self.btnkaydetClick)
        self.ui.btnguncelle.clicked.connect(self.btnguncelleClick)
        self.ui.tableWidget.clicked.connect(self.listeClick)
        self.ui.btnsil.clicked.connect(self.btnsilClick)


    def formYukle(self):
        self.ui.tableWidget.clear()
        self.ui.tableWidget.setColumnCount(4)
        self.ui.tableWidget.setHorizontalHeaderLabels(("ID", "İsim", "Soyisim", "Email"))
        self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        db = sql.connect("personelbilgisi.db")
        cur = db.cursor()
        sorgu = "Select * from Personeller"
        cur.execute(sorgu)

        satirlar = cur.fetchall()

        self.ui.tableWidget.setRowCount(len(satirlar))

        for satirIndeks, satirVeri in enumerate(satirlar):
            for sutunIndeks, sutunVeri in enumerate(satirVeri):
                self.ui.tableWidget.setItem(satirIndeks, sutunIndeks, QTableWidgetItem(str(sutunVeri)))

    def btnkaydetClick(self):
        id = self.ui.txtid.text()
        isim = self.ui.txtisim.text()
        soyisim = self.ui.txtsoyisim.text()
        email = self.ui.txtemail.text()

        try:
            self.baglanti = sql.connect("personelbilgisi.db")
            self.c = self.baglanti.cursor()
            self.c.execute("Insert into Personeller Values(?,?,?,?)",(id,isim,soyisim,email))
            self.baglanti.commit()
            self.c.close()
            self.baglanti.close()
            print("Basarili, Personel bilgisi veritabanina kaydedildi.")

        except Exception:
            print("Hata, Personel veritabanina kayit edilemedi")

        self.btnTemizle()
        self.formYukle()

    def btnTemizle(self):
        self.ui.txtid.clear()
        self.ui.txtisim.clear()  
        self.ui.txtsoyisim.clear()  
        self.ui.txtemail.clear() 

    def btnguncelleClick(self):
        id = self.ui.txtid.text()
        isim = self.ui.txtisim.text()
        soyisim = self.ui.txtsoyisim.text()
        email = self.ui.txtemail.text()     

        try:
            self.baglanti = sql.connect("personelbilgisi.db")
            self.c = self.baglanti.cursor()
            self.c.execute("update Personeller set isim = ?, soyisim = ?, email = ?",(isim,soyisim,email))
            self.baglanti.commit()
            self.c.close()
            self.baglanti.close()
            print("Basarili, Personel bilgisi veritabaninda guncellendi.")

        except Exception:
            print("Hata, Personel veritabaninada guncellenemedi.")

        self.btnTemizle()
        self.formYukle()

    def listeClick(self):
        self.ui.txtid.setText(self.ui.tableWidget.item(self.ui.tableWidget.currentRow(), 0).text())
        self.ui.txtisim.setText(self.ui.tableWidget.item(self.ui.tableWidget.currentRow(), 1).text()) 
        self.ui.txtsoyisim.setText(self.ui.tableWidget.item(self.ui.tableWidget.currentRow(), 2).text()) 
        self.ui.txtemail.setText(self.ui.tableWidget.item(self.ui.tableWidget.currentRow(), 3).text())      

    def btnsilClick(self):
        id = self.ui.txtid.text()

        try:
            self.baglanti = sql.connect("personelbilgisi.db")
            self.c = self.baglanti.cursor()
            self.c.execute("Delete from Personeller where id = ?", (id,))
            self.baglanti.commit()
            self.c.close()
            self.baglanti.close()
            print("Basarili, Personel bilgisi veritabanindan silindi.")

        except Exception:
            print("Hata, Personel veritabanindan silinemedi.")

        self.btnTemizle()
        self.formYukle()

def app():
    app =  QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())

app()
