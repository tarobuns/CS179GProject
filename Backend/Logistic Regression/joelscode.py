from pyspark.sql.functions import lit, udf
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql import SparkSession
from functools import reduce
import mysql.connector
import uuid
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

Node_Start = time.time()

# Define image dimensions
image_w, image_h = 224, 224

# Function to create a database if it doesn't exist
def create_database(cursor):
    try:
        cursor.execute("CREATE DATABASE IF NOT EXISTS trainedmodel;")
        print("Database trainedmodel created or already exists.")
    except mysql.connector.Error as err:
        print(f"Failed creating database: {err}")
        exit(1)

# Function to connect to MySQL and create the database if it doesn't exist
def connect_to_mysql():
    try:
        # Connect to MySQL server without specifying a database
        mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="toyota",
            port=3306
        )
        cursor = mydb.cursor()
        create_database(cursor)
        cursor.close()
        mydb.close()
        
        # Connect to the newly created database
        mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="toyota",
            port=3306,
            database="trainedmodel"
        )
        print("MySQL connection to trainedmodel successful")
        return mydb
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Connect to MySQL
mydb = connect_to_mysql()
if mydb is None:
    exit("Failed to connect to MySQL")

mycursor = mydb.cursor()

# Make the table if it does not exist
def create_table(cursor):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS Logistic_Regression (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model VARCHAR(255) NOT NULL,
        accuracy FLOAT NOT NULL,
        elapsed_time FLOAT NOT NULL,
        hdfs_path VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_sql)

# Create the table
create_table(mycursor)
mydb.commit()

img_dir = "hdfs://localhost:9000/toyota_cars"  # directory where cars are on hdfs
models = ["4runner", "alphard","avalon","avanza","avensis","aygo","camry","celica","corolla","corona","crown","estima","etios","fortuner","hiace","highlander","hilux","innova","iq","matrix","mirai","previa","prius","rav4","revo","rush","sequoia","sienna","soarer","starlet","supra","tacoma","tundra","venza","verso","vios","vitz","yaris"]
nb_classes = len(models)

spark = SparkSession.builder.appName('Toyota_Car') \
    .config('spark.ui.port', '4060') \
    .config('spark.executor.memory', '12g') \
    .config('spark.driver.memory', '12g') \
    .config('spark.executor.cores', '4') \
    .config('spark.executor.instances', '1') \
    .config('spark.sql.shuffle.partitions', '200') \
    .getOrCreate()  # this opens the spark session

# Load InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def process_image_file(row):
    try:
        img_data = row.data
        img_height = row.height
        img_width = row.width
        # Convert binary data to a NumPy array
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        # Reshape the NumPy array to the correct dimensions and convert BGR to RGB
        img_array = img_array.reshape((img_height, img_width, 3))[:, :, ::-1]
        img = Image.fromarray(img_array, 'RGB')
        img = img.resize((image_w, image_h))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except UnidentifiedImageError:
        return None
    except ValueError:
        return None

def extract_features(image_array):
    if image_array is not None:
        features = inception_model.predict(np.vstack([image_array]))
        return features.flatten().tolist()
    else:
        return None

extract_features_udf = udf(lambda img: extract_features(process_image_file(img)), ArrayType(FloatType()))

models_df = []
for idx, model in enumerate(models):
    car_df = spark.read.format("image").option("dropInvalid", True).load(img_dir + "/" + model).withColumn("label", lit(idx))
    car_df.printSchema()  # Debugging to check schema
    car_df.show(1)  # Show an example row for debugging
    car_df = car_df.withColumn("features", extract_features_udf(car_df["image"]))
    models_df.append(car_df.select("features", "label"))

def mergeDataframe():
    if models_df:
        return reduce(lambda x, y: x.union(y), models_df)
    else:
        return None

merged_df = mergeDataframe()

# print the dataframe to see if merge worked
if merged_df:
    merged_df = merged_df.filter(merged_df.features.isNotNull())
    merged_df = merged_df.withColumn("features", udf(lambda x: Vectors.dense(x), VectorUDT())(merged_df["features"]))
    merged_df.printSchema()
else:
    print("No dataframes to merge")

def split_data(df, train_ratio=0.7):
    """Splits the dataframe into training and testing sets."""
    return df.randomSplit([train_ratio, 1 - train_ratio])

train_df, test_df = split_data(merged_df)

def trainingModel(train_df, test_df, cursor, db_conn):
    try:
        lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

        # Create the pipeline
        pipeline = Pipeline(stages=[lr])

        start_time = time.time()  # Measure the time it takes
        model = pipeline.fit(train_df)
        end_time = time.time()
        elapsed_time = end_time - start_time

        predictions = model.transform(test_df)
        accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_df.count())

        unique_path = f"hdfs://localhost:9000/toyota_cars_predictions/{uuid.uuid4()}"  # This is a unique identifier for the path, so no two runs write to the same place
        # predictions.write.format("parquet").save(unique_path)
        model.write().overwrite().save(unique_path)

        # Log and store the results into the database
        sql = "INSERT INTO Logistic_Regression (model, accuracy, elapsed_time, hdfs_path) VALUES (%s, %s, %s, %s)"
        val = ("Logistic Regression", accuracy, elapsed_time, unique_path)
        print(f"Inserting into database: {val}")  # Debugging statement
        cursor.execute(sql, val)
        db_conn.commit()

        return model, accuracy, elapsed_time

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        db_conn.rollback()

# Call function to start training
trained_model, model_accuracy, training_time = trainingModel(train_df, test_df, mycursor, mydb)

# Query the table data after insertion
mycursor.execute("SELECT * FROM Logistic_Regression")
table_data = mycursor.fetchall()
print("\nTable Data After Insertion:")
for row in table_data:
    print(row)
 
Node_End = time.time()
total_exec_time = Node_Start - Node_End
print(f"Total execution time: {total_exec_time} seconds")

mycursor.close()
mydb.close()
spark.stop()
