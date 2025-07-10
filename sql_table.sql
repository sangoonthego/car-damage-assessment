CREATE DATABASE IF NOT EXISTS car_damage_db;
USE car_damage_db;

CREATE TABLE IF NOT EXISTS prediction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(255) NOT NULL,
    confidence FLOAT,
    predicted_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detection_logs (
	id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
	detected_class VARCHAR(255) NOT NULL,
    confidence FLOAT,
    x1 FLOAT,
    y1 FLOAT,
    x2 FLOAT,
    y2 FLOAT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS segmentation_logs (
	id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(255) NOT NULL,
    confidence FLOAT,
    segmentation_mask_path TEXT,
    predicted_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SELECT * FROM prediction_logs;
SELECT * FROM detection_logs;
SELECT * FROM segmentation_logs;




