from pyspark.sql.functions import lit, udf
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from functools import reduce
import mysql.connector
import time
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from pyspark.ml.linalg import Vectors
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
        print("MySQL connection to RF_MODEL successful")
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
    CREATE TABLE IF NOT EXISTS RF_training_results (
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


# Define image dimensions
image_w, image_h = 224, 224

img_dir = "hdfs://localhost:9000/toyota_cars"  # directory where cars are on hdfs
# Assign labels to images based on car models
models = ["4runner","alphard","avalon","avanza","avensis","aygo","camry","celica","corolla","corona","crown","estima","etios","fortuner","hiace","highlander","hilux","innova","iq","matrix","mirai","previa","prius","rav4","revo","rush","sequoia","sienna","soarer","starlet","supra","tacoma","tundra","venza","verso","vios","vitz","yaris"]  # Start with a smaller subset of models to test
models_sample = ["alphard","avanza"]  # Start with a smaller subset of models to test

nb_classes = len(models)
nb_classes_sampe = len(models_sample)

X, Y = [], []
models_df = []  # Define models_df before using it
spark = SparkSession.builder.appName('Toyota_Car').getOrCreate()  # this opens the spark session

# Load InceptionV3 model
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def process_image_files(files):
    processed_images = []
    for file in files:
        try:
            img_data = file['image']['data']
            img_height = file['image']['height']
            img_width = file['image']['width']
            # Convert binary data to a NumPy array
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            # Reshape the NumPy array to the correct dimensions and convert BGR to RGB
            img_array = img_array.reshape((img_height, img_width, 3))[:, :, ::-1]
            img = Image.fromarray(img_array, 'RGB')
            img = img.resize((image_w, image_h))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            processed_images.append(img_array)
        except UnidentifiedImageError:
            print(f"UnidentifiedImageError for image: {file['image']['origin']}")
        except ValueError:
            print(f"ValueError for image: {file['image']['origin']}")
    return processed_images

def extract_features(images_batch):
    # features = inception_model.predict(np.vstack(images_batch))
    features = inception_model.predict(np.vstack(images_batch))
    return features

def process_image_udf(image_data):
    try:
        img_array = np.frombuffer(image_data, dtype=np.uint8).reshape((image_h, image_w, 3))[:, :, ::-1]
        img = image.array_to_img(img_array)
        img = img.resize((image_w, image_h))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = inception_model.predict(img_array).flatten()
        return features.tolist()
    except Exception as e:
        print(f"Error processing image: {e}")
        return []
    
# Register the UDF
process_image = udf(process_image_udf, ArrayType(FloatType()))

# Define a UDF to convert arrays to dense vectors
def array_to_vector_udf(array):
    return Vectors.dense(array)

array_to_vector = udf(array_to_vector_udf, VectorUDT())
# def makeRdd(model, number):
#     car_df = spark.read.format("image").option("dropInvalid", True).load(img_dir + "/" + model).withColumn("label", lit(number))  # Reading image of folder
#     # car_df = car_df.limit(100)  # Limit to 100 images per model for initial testing
#     # models_df.append(car_df)  # appending the data frame to the list
#     # files = car_df.collect()
#     # for i in range(0, len(files)):
#     #     batch_files = files[i:i+batch_size]
#     #     processed_images = process_image_files(batch_files)
#     processed_images = process_image_files(car_df)
#     if processed_images:
#         features = extract_features(processed_images)
#         for feature in features:
#             X.append(feature.flatten().tolist())
#             Y.append(number)
#     # a = car_df.count()
#     # car_df.printSchema()
#     # print(f"rows: {a}")
#     # car_df.show()

def makeRdd(model, number):
    car_df = spark.read.format("image").option("dropInvalid", True).load(f"{img_dir}/{model}").withColumn("label", lit(number)).limit(150)
    models_df.append(car_df)
    car_df = car_df.withColumn('features_array', process_image(car_df['image']['data']))
    car_df = car_df.withColumn('features', array_to_vector(car_df['features_array']))
    car_df = car_df.select('features', 'label')
    return car_df

for idx, model in enumerate(models):
    makeRdd(model, idx)

# Create a new DataFrame with the features and labels
features_df = spark.createDataFrame([(Vectors.dense(x), int(y)) for x, y in zip(X, Y)], ["features", "label"])

def split_data(df, train_ratio=0.7):
    """Splits the dataframe into training and testing sets."""
    return df.randomSplit([train_ratio, 1 - train_ratio])

train_df, test_df = split_data(features_df)
# train_df = features_df

def trainingModel(train_df, test_df,numOfTree):
    # try:
        start_time = time.time()  # Measure the time it takes
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=numOfTree)
        pipeline = Pipeline(stages=[rf])
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)
        predictions.select("probability","prediction", "label", "features").show(5)
        evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Test Error = %g" % (1.0 - accuracy))
        end_time = time.time()
        elapsed_time = end_time - start_time
        # unique_path = "/slee809/RFmodel" 
        unique_path = f"/toyota_cars_predictions_RF/RF_model_numberoftree_{numOfTree}"  # This is a unique identifier for the path, so no two runs write to the same place
        model.write().overwrite().save(unique_path)
        sql = "INSERT INTO RF_training_results (model, accuracy, elapsed_time, hdfs_path) VALUES (%s, %s, %s, %s)"
        val = ("Random Forest", accuracy, elapsed_time, unique_path)
        print(f"Inserting into database: {val}")  # Debugging statement
        mycursor.execute(sql, val)
        mydb.commit()
        return model, accuracy, elapsed_time

ac = []
# Call function to start training
trained_model, model_accuracy, training_time = trainingModel(train_df,test_df,25)
ac.append(model_accuracy)
trained_model, model_accuracy, training_time = trainingModel(train_df,test_df,50)
ac.append(model_accuracy)
trained_model, model_accuracy, training_time = trainingModel(train_df,test_df,100)
ac.append(model_accuracy)
trained_model, model_accuracy, training_time = trainingModel(train_df,test_df,150)
ac.append(model_accuracy)
print(ac)

print("End")
print(training_time)
spark.stop()
