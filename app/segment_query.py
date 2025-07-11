import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from datetime import datetime
from app.database import Database

class SegmentationLogManager:
    def __init__(self):
        pass

    def save_segmentation_log(self, img_name, segmentations):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor()
        try:
            query = """
                INSERT INTO segmentation_logs (image_name, predicted_class, confidence, severity, segmentation_mask_path, predicted_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            for seg in segmentations:
                values = (
                    img_name,
                    seg["class"],
                    seg["confidence"],
                    seg["severity"],
                    seg["mask_path"],
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

    def get_segmentation(self):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor(dictionary=True)
        try:
            query = "SELECT * FROM segmentation_logs ORDER BY predicted_at DESC"
            cursor.execute(query)
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def get_segmentatation_severity(self, predicted_class, severity):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor(dictionary=True)

        try:
            query = "" \
            "SELECT * FROM segmentation_logs WHERE LOWER(predicted_class) = %s " \
            "AND LOWER(severity) = %s"

            cursor.execute(query, (predicted_class.strip(), severity,))
            logs = cursor.fetchall()
            return logs
        except Exception as e:
            logging.error(f"SQL Fail: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def delete_segmentation(self, id):
        db = Database()
        conn = db.connect()
        cursor = conn.cursor()
        try:
            query = "DELETE FROM segmentation_logs WHERE id = %s"
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
