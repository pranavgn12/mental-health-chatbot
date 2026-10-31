from database import engine

try:

    conn = engine.connect()

    print("Connected Successfully")

    conn.close()

except Exception as e:

    print("Connection Failed")

