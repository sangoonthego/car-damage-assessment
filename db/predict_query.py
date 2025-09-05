import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from datetime import datetime
from pydantic import BaseModel

from db.database import Database

class PredictionLogManager:
    def __init__(self):
        pass

    def save_prediction_log(self, img_name, pred_class, confidence):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor()
        try:
            query = """
                INSERT INTO prediction_logs (image_name, predicted_class, confidence, predicted_at)
                VALUES (%s, %s, %s, %s)
            """
            values = (img_name, pred_class, confidence, datetime.now())
            cursor.execute(query, values)
            conn.commit()
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
        finally:
            cursor.close()
            conn.close()

    def get_prediction(self):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor(dictionary=True)
        try:
            query = "SELECT * FROM prediction_logs ORDER BY predicted_at DESC"
            cursor.execute(query)
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return False
        finally:
            cursor.close()
            conn.close()

    def get_best_confidence(self, confidence):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor(dictionary=True)
        try:
            query = (
                "SELECT * FROM prediction_logs "
                "WHERE confidence > %s "
                "ORDER BY confidence DESC"
            )
            cursor.execute(query, (confidence,))
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_prediction_label(self, pred_class):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor(dictionary=True)
        try:
            pred_class = pred_class.strip()
            query = (
                "SELECT * FROM prediction_logs "
                "WHERE LOWER(predicted_class) = LOWER(%s)"
            )
            cursor.execute(query, (pred_class,))
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_best_prediction_label(self, pred_class, confidence):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor(dictionary=True)
        try:
            pred_class = pred_class.strip()
            query = (
                "SELECT * FROM prediction_logs "
                "WHERE LOWER(predicted_class) = LOWER(%s) AND confidence > %s"
            )
            cursor.execute(query, (pred_class, confidence))
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_prediction_image(self, img_name):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor(dictionary=True)
        try:
            img_name = img_name.strip()
            query = (
                "SELECT * FROM prediction_logs "
                "WHERE LOWER(image_name) = LOWER(%s)"
            )
            cursor.execute(query, (img_name,))
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def update_prediction(self, id, corrected_label):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor()
        try:
            query = (
                "UPDATE prediction_logs SET corrected_label = %s WHERE id = %s"
            )
            cursor.execute(query, (corrected_label, id))
            affected = cursor.rowcount
            conn.commit()
            return affected > 0
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def delete_uncorrect_prediction(self, id):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor()
        try:
            query = (
                "DELETE FROM prediction_logs "
                "WHERE id = %s"
            )
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

class UpdateLog(BaseModel):
    corrected_label: str
