from pyspark.ml.pipeline import PipelineModel
from PIL import Image
import numpy as np 
from tensorflow.keras.applications.inception_v3 import preprocess_input
from pyspark.sql import SparkSession
import glob
from pyspark.ml.linalg import Vectors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
# Define constants
caltech_dir = "/home/cs179g/code/Pyo/sample"
image_w = 224
image_h = 224
pixels = image_h * image_w * 3
models = ["4runner","alphard","avalon","avanza","avensis","aygo","camry","celica","corolla","corona","crown","estima","etios","fortuner","hiace","highlander","hilux","innova","iq","matrix","mirai","previa","prius","rav4","revo","rush","sequoia","sienna","soarer","starlet","supra","tacoma","tundra","venza","verso","vios","vitz","yaris"]

# Initialize empty lists
X = []
Y = []
tmp =[]
filenames = []

spark = SparkSession.builder.appName('Toyota_Car').getOrCreate()  # this opens the spark session
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def extract_features(images_batch):
    features = inception_model.predict(np.vstack(images_batch))
    return features
# Load images from the directory
files = glob.glob(caltech_dir + "/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    tmp.append(img_array)

features = extract_features(tmp)
for feature in features:
    X.append(feature.flatten().tolist())

# img_array = img_array.flatten().tolist()  # Flatten the array to match expected input format
# filenames.append(f)
# X.append((Vectors.dense(img_array),))

# for item in X:
#     print(type(item[0]), item[0])

# Convert X to a RDD
# features_df = spark.createDataFrame([Vectors.dense(zip(X,['0','1']))],["features"])

test_df = spark.createDataFrame([(Vectors.dense(x),) for x in X], ["features"])

# model_path = "/slee809/RFmodel" 
model_path = "hdfs://localhost:9000/toyota_cars_dt_predictions/fbf9f767-c96f-4d4d-beac-05fc71e2357a" 

# Load the model
persistedModel = PipelineModel.load(model_path)

# Make predictions
predictions = persistedModel.transform(test_df)
predictions.show()

prediction_list = predictions.select("prediction").collect()
probability_list = predictions.select("probability").collect()
for pred in prediction_list:
    print(models[int(pred.prediction)])
for prob in probability_list:
    print(max(prob.probability))