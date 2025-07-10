import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datetime import datetime
import logging
from app.database import connect_db

def save_detection_log(img_name, detections):
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO detection_logs (image_name, detected_class, confidence, x1, y1, x2, y2, detected_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        for detection in detections:
            values = (
                img_name,
                detection["class"],
                detection["confidence"],
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
                datetime.now()
            )

            cursor.execute(query, values)

        conn.commit()
    
    except Exception as e:
        logging.error(f"SQL Fail: {e}")
        return None
    
    finally:
        cursor.close()
        conn.close()

def get_detection():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    try:
        query = "" \
        "SELECT * FROM detection_logs " \
        "ORDER BY detected_at DESC"

        cursor.execute(query)
        logs = cursor.fetchall()

        return logs
    
    except Exception as e:
        logging.error(f"SQL Fail: {e}")
        return None

    finally:
        cursor.close()
        conn.close()

def get_detection_label(detected_class):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    try:
        detected_class = detected_class.strip()
        query = "" \
        "SELECT * FROM detection_logs " \
        "WHERE LOWER(detected_class) = LOWER(%s)"

        cursor.execute(query, (detected_class,))
        logs = cursor.fetchall()

        return logs
    
    except Exception as e:
        logging.error(f"SQL Fail: {e}")
        return None
    
    finally:
        cursor.close()
        conn.close()

def get_detection_image(img_name):
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    try:
        img_name = img_name.strip()
        query = "" \
        "SELECT * FROM detection_logs " \
        "WHERE LOWER(img_name) = LOWER(&s)"

        cursor.execute(query, (img_name,))
        logs = cursor.fetchall()

        return logs
    
    except Exception as e:
        logging.error(f"SQL Fail: {e}")
        return None
    
    finally:
        cursor.close()
        conn.close()

def delete_detection(id):
    conn = connect_db()
    cursor = conn.cursor()

    try:
        query = "" \
        "DELETE FROM detection_logs WHERE id = &s"

        cursor.execute(query, (id,))
        affected = cursor.rowcount
        conn.commit()

        return affected > 0
    
    except Exception as e:
        logging.error(f"SQL Fail: {e}")
        return None
    
    finally:
        cursor.close()
        conn.close()