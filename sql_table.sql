CREATE DATABASE IF NOT EXISTS car_damage_db;
USE car_damage_db;

CREATE TABLE IF NOT EXISTS prediction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_name VARCHAR(255) NOT NULL,
    predicted_class VARCHAR(255) NOT NULL,
    confidence FLOAT,
    predicted_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

SELECT * FROM prediction_logs;

