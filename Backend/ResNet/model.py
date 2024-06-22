import numpy as np
import pandas as pd
import tensorflow as tf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, udf
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType, BinaryType
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from PIL import Image, ImageDraw, UnidentifiedImageError
import sys
from pyspark.sql.functions import substring_index, split
from typing import Iterator

# spark-submit --master local[*] --driver-memory 12g --executor-memory 12g --conf spark.ui.port=5050 model.py

# # Configure Spark session, avoid java heap space error
# spark = SparkSession.builder \
#     .appName("CarImageClassification") \
#     .config("spark.driver.memory", "12g") \
#     .config("spark.executor.memory", "12g") \
#     .config("spark.ui.port", "5050")

# Initialize a Spark session
print("Initializing Spark session...")
spark = SparkSession.builder.appName("CarImageClassification").getOrCreate()

# # Define UDF to convert images to arrays
# def image_to_array(image):
#     try:
#         img = Image.open(image)
#         img = img.resize((224, 224))
#         np_img = np.asarray(img)
#         print(np_img.shape)
#         print(np_img)
#         return np_img.flatten().tolist()
#     except Exception as e:
#         print("Error in image_to_array: ", e)
#         return

# image_to_array_udf = udf(image_to_array, ArrayType(FloatType()))

# Load images from a directory
print("Loading images from directory...")
image_df = spark.read.format("image").load("toyota_cars/*").filter("image.nChannels > 2 AND image.height < 700")

# image_row = 40
# spark_single_img = image_df.select("image").collect()[image_row]
# (spark_single_img.image.origin, spark_single_img.image.mode, spark_single_img.image.nChannels)

# mode = 'RGBA' if (spark_single_img.image.nChannels == 4) else 'RGB' 
# Image.frombytes(mode=mode, data=bytes(spark_single_img.image.data), size=[spark_single_img.image.width,spark_single_img.image.height]).show()

# # Convert an image
# def convert_bgr_array_to_rgb_array(img_array):
#     B, G, R = img_array.T
#     return np.array((R, G, B)).T

# img = Image.frombytes(mode=mode, data=bytes(spark_single_img.image.data), size=[spark_single_img.image.width,spark_single_img.image.height])

# converted_img_array = convert_bgr_array_to_rgb_array(np.asarray(img))
# Image.fromarray(converted_img_array).show()

# # Resize all images
# schema = StructType(image_df.select("image.*").schema.fields + [
#     StructField("data_as_resized_array", ArrayType(IntegerType()), True),
#     StructField("data_as_array", ArrayType(IntegerType()), True)
# ])

# def resize_img(img_data, resize=True):
#     mode = 'RGBA' if (img_data.nChannels == 4) else 'RGB' 
#     img = Image.frombytes(mode=mode, data=img_data.data, size=[img_data.width, img_data.height])
#     img = img.convert('RGB') if (mode == 'RGBA') else img
#     img = img.resize([224, 224], resample=Image.Resampling.BICUBIC) if (resize) else img
#     arr = convert_bgr_array_to_rgb_array(np.asarray(img))
#     arr = arr.reshape([224*224*3]) if (resize) else arr.reshape([img_data.width*img_data.height*3])

#     return arr

# def resize_image_udf(dataframe_batch_iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
#     for dataframe_batch in dataframe_batch_iterator:
#         dataframe_batch["data_as_resized_array"] = dataframe_batch.apply(resize_img, args=(True,), axis=1)
#         dataframe_batch["data_as_array"] = dataframe_batch.apply(resize_img, args=(False,), axis=1)
#         yield dataframe_batch

# resized_df = image_df.select("image.*").mapInPandas(resize_image_udf, schema)

# # Use ResNet50 to predict
# def normalize_array(arr):
#     return tf.keras.applications.resnet50.preprocess_input(arr.reshape([224,224,3]))

# @pandas_udf(ArrayType(FloatType()))
# def predict_batch_udf(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
#     model = ResNet50()
#     for input_array in iterator:
#         normalized_input = np.stack(input_array.map(normalize_array))
#         preds = model.predict(normalized_input)
#         yield pd.Series(list(preds))

# predicted_df = resized_df.withColumn("predictions", predict_batch_udf("data_as_resized_array"))

#Output a prediction

# sys.exit()

# # Apply UDF
# print("Applying UDF to convert images to arrays...")
# image_df = image_df.withColumn("image_array", image_to_array_udf(col("image.data")))

# Check the schema to see what columns are present
print("DataFrame Schema before label/image_array:")
image_df.printSchema()

# Create new column, "label" with the car models
#print(image_df.select(col("image.origin")).show())
image_df = image_df.withColumn("label", split(image_df["image"]["origin"], "/")[8])
#print(image_df.select("label").show())

# Borrowed from Joel's code
def process_image(file):
        try:
            img_data = file["data"]
            img_height = file["height"]
            img_width = file["width"]
            # Convert binary data to a NumPy array
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            # Reshape the NumPy array to the correct dimensions and convert BGR to RGB
            #img_array = img_array.reshape((img_height, img_width, 3))[:, :, ::-1]
            
            return img_array
            img = Image.fromarray(img_array, 'RGB')
            img = img.resize((224, 224))
            img_array = Image.image.img_to_array(img) # Convert PIL Image to Numpy Array
            img_array = np.expand_dims(img_array, axis=0)
            print(img_array.shape)
            return img_array
        except UnidentifiedImageError as e:
            print("UnidentifiedImageError in process_image: ", e)
        except ValueError as e:
            print("ValueError in process_image: ", e)
        except TypeError as e:
            print("TypeError in process_image: ", e)

# Define UDF to apply the process_image function to each row of the DataFrame
process_image_udf = udf(process_image, ArrayType(FloatType()))

# Apply the UDF to the DataFrame to create a new column of processed images
image_df = image_df.withColumn("processed_images", process_image_udf("image"))

# Check the schema to see what columns are present after label creation
print("DataFrame Schema after label/image_array:")
image_df.printSchema()

#image_df.select("processed_images").show(10)

# print(image_df.select(split(image_df["image"]["origin"], "/")[-2]).distinct().show())
# print(image_df.select("label").distinct().show())
#sys.exit()

# Convert to Pandas DataFrame
print("Converting Spark DataFrame to Pandas DataFrame...")
pandas_df = image_df.select("processed_images", "label").toPandas()

# Extract features and labels
print("Extracting features and labels...")
X = np.array(pandas_df["processed_images"].tolist())
y = pd.get_dummies(pandas_df["label"]).values

# Check the size of the array
print("Original shape:", X.shape)

# Reshape X to the required format
print("Reshaping features...")
try:
    X = X.reshape(-1, 224, 224, 3)
except Exception as e:
    print("Error during reshaping: ", e)

# Split data into training and validation sets
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ResNet model
print("Defining ResNet model...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(y.shape[1], activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
print("Freezing base model layers...")
for layer in base_model.layers:
    layer.trainable = False

# Compile model
print("Compiling model...")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_resnet_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the trained model
print("Saving the trained model...")
model.save('resnet50_car_classifier.h5')

print("Training completed and model saved successfully.")