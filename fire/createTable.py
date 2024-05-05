import sqlite3 as sql

def main():
    try:
        db = sql.connect("personelbilgisi.db")
        cur = db.cursor()
        sorgu = "Create Table Personeller (id int, isim txt, soyisim txt, email txt)"

        cur.execute(sorgu)
        print("tablo olusturulmustur.")
    except sql.error as e:
        print("Tablo olusturulurken bir hata olustu.")
    finally:
        db.close()

if __name__ == "__main__":
    main()