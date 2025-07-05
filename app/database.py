import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Knkidngoc2005!",
        database="car_damage_db"
    )
