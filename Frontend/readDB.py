from flask import Flask, jsonify, render_template
import mysql.connector

app = Flask(__name__)

# Function to connect to MySQL database and execute a SELECT statement
def fetch_data_from_mysql():
    try:
        # Establish the connection
        mydb = mysql.connector.connect(
            host="127.0.0.1",  # Replace with your MySQL server host
            user="root",       # Replace with your MySQL username
            password="toyota", # Replace with your MySQL password
            port=3306,
            database="trainedmodel"  # Replace with your database name
        )

        # Create a cursor object
        mycursor = mydb.cursor()
        # Execute a simple SELECT statement
        mycursor.execute("SELECT * FROM RF_training_results")  # Replace with your table name
        # Fetch all the rows
        result = mycursor.fetchall()
        # Convert result to list of dictionaries
        columns = [desc[0] for desc in mycursor.description]
        data1 = [dict(zip(columns, row)) for row in result]
        
        mycursor.execute("SELECT * FROM Logistic_Regression")  # Replace with your table name
        result = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        # data2 = [dict(zip(columns, row)) for row in result]
        data2 = [dict(zip(columns, row)) for row in result]
        final_data = data1+data2

        ####Varun's part
        mycursor.execute("SELECT * FROM dec_tree") # Replace with your table name
        result = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        # data2 = [dict(zip(columns, row)) for row in result]
        data3 = [dict(zip(columns, row)) for row in result]
        #####  
        final_data = data1+data2+data3
        return final_data

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

    finally:
        # Close the cursor and connection
        if mycursor:
            mycursor.close()
        if mydb:
            mydb.close()

def getNameAndHDFSurl(index):
    try:
        # Establish the connection
        mydb = mysql.connector.connect(
            host="127.0.0.1",  # Replace with your MySQL server host
            user="root",       # Replace with your MySQL username
            password="toyota", # Replace with your MySQL password
            port=3306,
            database="trainedmodel"  # Replace with your database name
        )

        # Create a cursor object
        mycursor = mydb.cursor()
        # Execute a simple SELECT statement
        mycursor.execute("SELECT * FROM RF_training_results")  # Replace with your table name
        # Fetch all the rows
        result = mycursor.fetchall()
        # Convert result to list of dictionaries
        columns = [desc[0] for desc in mycursor.description]
        data1 = [dict(zip(columns, row)) for row in result]
        
        mycursor.execute("SELECT * FROM Logistic_Regression")  # Replace with your table name
        result = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        # data2 = [dict(zip(columns, row)) for row in result]
        data2 = [dict(zip(columns, row)) for row in result]

        ##Varun's Part
        mycursor.execute("SELECT * FROM dec_tree") # Replace with your table name
        result = mycursor.fetchall()
        columns = [desc[0] for desc in mycursor.description]
        # data2 = [dict(zip(columns, row)) for row in result]
        data3 = [dict(zip(columns, row)) for row in result]
        ######

        final_data = data1+data2 +data3      #data3
        
        path = final_data[index-1]['hdfs_path']
        model = final_data[index-1]['model']
        
        return path,model

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return []

    finally:
        # Close the cursor and connection
        if mycursor:
            mycursor.close()
        if mydb:
            mydb.close()


# @app.route('/models')
# def index():
#     return render_template('models.html')

# @app.route('/api')
# def get_data():
#     # Fetch data from MySQL and return as JSON
#     data = fetch_data_from_mysql()
#     return jsonify(data)

# @app.route('/remote-connect')
# def remote_connect():
#     # Specify your remote server details here
#     hostname = '169.235.30.69'
#     port = 22  # Default SSH port, change if yours is different
#     username = 'vnara009'
#     password = 'xxx'
#     # Connect to the remote server and return status
#     connection_status = connect_remote_server(hostname, port, username, password)
#     return connection_status