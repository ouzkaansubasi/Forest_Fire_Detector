import sqlite3 as sql

def main():
    try:
        db = sql.connect("personelbilgisi.db")
        print("Veritabani olusturulmustur.")
    except:
        print("Veritabanina baglanma hatasi gerceklesti.")
    finally:
        db.close()

if __name__ == "__main__":
    main()