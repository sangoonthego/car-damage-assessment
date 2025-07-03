import mysql.connector
from datetime import datetime

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Knkidngoc2005!",
        database="car_damage_db"
    )

def save_prediction_log(img_name, pred_class, confidence):
    conn = connect_db()
    cursor = conn.cursor()
    query = """
        INSERT INTO prediction_logs (image_name, predicted_class, confidence, predicted_at)
        VALUES (%s, %s, %s, %s)
    """
    values = (img_name, pred_class, confidence, datetime.now())
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()