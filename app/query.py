import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
from pydantic import BaseModel

from app.database import connect_db

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

def get_prediction():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    try:
        query = "" \
        "SELECT * FROM prediction_logs ORDER BY predicted_at DESC"

        cursor.execute(query)
        logs = cursor.fetchall()

        return logs

    except Exception as e:
        print(f"SQL Fail: {e}")
        return False

    finally:
        cursor.close()
        conn.close()

def get_prediction_label(pred_class):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    try:
        pred_class = pred_class.strip()
        query = "" \
        "SELECT * FROM prediction_logs WHERE LOWER(predicted_class) = LOWER(%s)"

        cursor.execute(query, (pred_class,))
        logs = cursor.fetchall()

        return logs

    except Exception as e:
        print(f"SQL Fail: {e}")

    finally:
        cursor.close()
        conn.close()

def get_best_prediction_label(pre_class, confidence):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    try:
        pred_class = pred_class.strip()
        query = "" \
        "SELECT * FROM prediction_logs WHERE LOWER(predicted_class) = LOWER(%s) AND confidence > 0.9"

        cursor.execute(query, (pre_class, confidence,))
        logs = cursor.fetchall()
        
        return logs
    
    except Exception as e:
        print(f"SQL Fail: {e}")

    finally:
        cursor.close()
        conn.close()


def get_prediction_image(img_name):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    try:
        img_name = img_name.strip()
        query = "" \
        "SELECT * FROM prediction_logs WHERE LOWER(image_name) = LOWER(%s)"

        cursor.execute(query, (img_name,))
        logs = cursor.fetchall()

        return logs
    
    except Exception as e:
        print(f"SQL Fail: {e}")

    finally:
        cursor.close()
        conn.close()

class UpdateLog(BaseModel):
    corrected_label: str

def update_prediction(id, corrected_label):
    conn = connect_db()
    cursor = conn.cursor()

    try:
        query = "" \
        "UPDATE prediction_logs SET corrected_label = %s WHERE id = %s"

        cursor.execute(query, (corrected_label, id,))
        affected = cursor.rowcount
        conn.commit()

        return affected > 0
    
    except Exception as e:
        print(f"SQL Fail: {e}")

    finally:
        cursor.close()
        conn.close()
    
def delete_uncorrect_prediction(id):
    conn = connect_db()
    cursor = conn.cursor()

    try:   
        query = "" \
        "DELETE FROM prediction_logs WHERE id = %s"

        cursor.execute(query, (id,))
        affected = cursor.rowcount
        conn.commit()

        return affected > 0
    
    except Exception as e:
        print(f"SQL Fail: {e}")

    finally:
        cursor.close()
        conn.close()

