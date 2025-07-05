import mysql.connector
from datetime import datetime
from pydantic import BaseModel

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Knkidngoc2005!",
        database="car_damage_db"
    )
