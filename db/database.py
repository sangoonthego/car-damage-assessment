import mysql.connector

class Database:
    def __init__(self, host="localhost", user="root", password="Knkidngoc2005!", database="car_damage_db"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
