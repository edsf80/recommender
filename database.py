import psycopg2

def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(host="localhost",database="recommender", user="postgres", password="secret")
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)