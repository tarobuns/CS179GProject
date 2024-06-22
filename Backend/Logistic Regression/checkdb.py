import mysql.connector

# Connect to the MySQL database
mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="toyota",
    port=3306,
    database="VectorImage"
)

# Create a cursor object
mycursor = mydb.cursor()

# Execute a SELECT query to retrieve all rows from the training_results table
mycursor.execute("SELECT * FROM training_results")

# Fetch all the rows returned by the query
table_data = mycursor.fetchall()

# Print the table data
print("Table Data:")
for row in table_data:
    print(row)

# Close the cursor and database connection
mycursor.close()
mydb.close()
