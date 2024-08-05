import psycopg2

class DatabaseConnection:
    def __init__(self, dbname, user, password, host, port):
        self.conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        self.cur = self.conn.cursor()
    
    def query_number(self, number):
        query = "SELECT num FROM bus_number WHERE num = %s;"
        self.cur.execute(query, (number,))
        return self.cur.fetchone()
    
    def close(self):
        self.cur.close()
        self.conn.close()
