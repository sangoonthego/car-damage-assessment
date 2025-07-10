import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from datetime import datetime
from app.database import connect_db

def save_segmentation_log(img_name, segmentations):
    conn = connect_db()
    cursor = conn.cursor()

    try:

        query = """
            INSERT INTO segmentation_logs (image_name, predicted_class, confidence, segmentation_mask_path, predicted_at)
            VALUES (%s, %s, %s, %s, %s)
        """

        for seg in segmentations:
            values = (
                img_name,
                seg["class"],
                seg["confidence"],
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

    