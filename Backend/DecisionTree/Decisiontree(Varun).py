from pyspark.sql.functions import lit, udf
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, FloatType
from pyspark.ml.linalg import Vectors, VectorUDT
from functools import reduce
import mysql.connector
import uuid
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image

# Set image dimensions
image_w, image_h = 224, 224


# Function to establish a connection to MySQL and to the database
def connect_to_mysql():
    try:  
        mydb = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="toyota",
            port=3306,
            database="trainedmodel"
        )
        print("Successfully connected to trainedmodel database")
        return mydb
    except mysql.connector.Error as err:
        print(f"Connection error: {err}")
        return None

# Establish a connection to MySQL
mydb = connect_to_mysql()
if not mydb:
    exit("Failed to connect to MySQL")

mycursor = mydb.cursor()

# Function to create the table if it doesn't exist
def create_table(cursor):
    table_creation_query = """
    CREATE TABLE IF NOT EXISTS dec_tree (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model VARCHAR(255) NOT NULL,
        accuracy FLOAT NOT NULL,
        elapsed_time FLOAT NOT NULL,
        hdfs_path VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(table_creation_query)

# Create the table
create_table(mycursor)
mydb.commit()

img_dir = "hdfs://localhost:9000/toyota_cars"  # HDFS directory containing car images
#The car models list. When training I did use a permutation and combination of these models plus modified the hyper-parameters in the model. Hence the various models.
models = ["4runner","alphard","avalon","avanza","avensis","aygo","camry","celica","corolla","corona","crown","estima","etios","fortuner","hiace","highlander","hilux","innova","iq","matrix","mirai","previa","prius","rav4","revo","rush","sequoia","sienna","soarer","starlet","supra","tacoma","tundra","venza","verso","vios","vitz","yaris"]
models_df = []  # List to store dataframes

# Initialize Spark session with specified configurations
spark = SparkSession.builder \
    .appName('Toyota_Car') \ 
    .config('spark.ui.port', '4050') \  
    .config('spark.executor.memory', '12g') \ 
    .config('spark.executor.instances', '2') \  
    .config('spark.sql.shuffle.partitions', '150') \  
    .getOrCreate()


# Loading of the InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Define a UDF to process images and extract features
def process_image_udf(image_data):
    try:
        img_array = np.frombuffer(image_data, dtype=np.uint8).reshape((image_h, image_w, 3))[:, :, ::-1]
        img = image.array_to_img(img_array).resize((image_w, image_h))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        features = inception_model.predict(img_array).flatten()
        return features.tolist()
    except Exception as e:
        print(f"There has been an error proccessing the image {e}")
        emptylist=[]
        return emptylist


# Register the UDF
process_image = udf(process_image_udf, ArrayType(FloatType()))

# Define a UDF to convert arrays to dense vectors
def array_to_vector_udf(array):
    return Vectors.DenseVector(array)

array_to_vector = udf(array_to_vector_udf, VectorUDT())

# Read images and process using the UDF
def makeRdd(model, number):
    car_df = spark.read.format("image").load(f"{img_dir}/{model}").option("dropInvalid", True).limit(150)
    models_df.append(car_df)
    car_df = car_df.withColumn("label", lit(number))
    car_df = car_df.withColumn('features_array', process_image(car_df['image']['data']))
    car_df = car_df.withColumn('features', array_to_vector(car_df['features_array']))
    car_df = car_df.select('features', 'label')
    return car_df


# Process each car model
for idx, model in enumerate(models):
    car_df = makeRdd(model, idx)
    if car_df:
        if 'features_df' in locals():
            features_df = features_df.union(car_df)
        else:
            features_df = car_df


# Function to merge dataframes
def mergeDataframe():
    return reduce(lambda x, y: x.union(y), models_df) if models_df else None

merged_df = mergeDataframe()
if mergeDataframe():
    merged_df.printSchema()
else:
    print("No dataframes to merge")

# Function to split data into training and testing sets
def split_data(df, train_ratio=0.7):
    return df.randomSplit([train_ratio, 0.3])

train_df, test_df = split_data(features_df)

# Function to train the model and log the results to the database
def trainingModel(train_df, test_df, cursor, db_conn):
    try:
        dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=5)
        pipeline = Pipeline(stages=[dt])
        start_time = time.time()
        model = pipeline.fit(train_df)
        end_time=time.time()
        elapsed_time = end_time - start_time

        predictions = model.transform(test_df)
        accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test_df.count())

        unique_path = f"hdfs://localhost:9000/toyota_cars_dt_predictions/{uuid.uuid4()}"
        model.write().overwrite().save(unique_path)

        sql = "INSERT INTO dec_tree (model, accuracy, elapsed_time, hdfs_path) VALUES (%s, %s, %s, %s)"
        val = ("Decision Tree", accuracy, elapsed_time, unique_path)
        cursor.execute(sql, val)
        db_conn.commit()

        return model, accuracy, elapsed_time

    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        db_conn.rollback()

# Train the model and log results
trained_model, model_accuracy, training_time = trainingModel(train_df, test_df, mycursor, mydb)

# Query the table to display the data after insertion
mycursor.execute("SELECT * FROM dec_tree")
for row in mycursor.fetchall():
    print(row)

mycursor.close()
mydb.close()
spark.stop()
