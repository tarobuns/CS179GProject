from pyspark.ml.pipeline import PipelineModel
from PIL import Image
import numpy as np 
from pyspark.sql import SparkSession
import glob
from pyspark.ml.linalg import Vectors
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input 
from readDB import getNameAndHDFSurl
# Define constants
def Predict(img,index):
        path, _ = getNameAndHDFSurl(index)
        image_w = 224
        image_h = 224
        models = ["4runner","alphard","avalon","avanza","avensis","aygo","camry","celica","corolla","corona","crown","estima","etios","fortuner","hiace","highlander","hilux","innova","iq","matrix","mirai","previa","prius","rav4","revo","rush","sequoia","sienna","soarer","starlet","supra","tacoma","tundra","venza","verso","vios","vitz","yaris"]
        # Initialize empty lists
        X = []
        tmp =[]
        spark = SparkSession.builder.appName('Toyota_Car').getOrCreate()  # this opens the spark session
        inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

        def extract_features(images_batch):
            features = inception_model.predict(np.vstack(images_batch))
            return features
        
        img = img.convert("RGB")
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        tmp.append(img_array)
        features = extract_features(tmp)
        for feature in features:
            X.append(feature.flatten().tolist())
        test_df = spark.createDataFrame([(Vectors.dense(x),) for x in X], ["features"])
        model_path = path 
        # Load the model
        persistedModel = PipelineModel.load(model_path)
        # Make predictions
        predictions = persistedModel.transform(test_df)
        predictions.show()
        prediction_list = predictions.select("prediction").collect()
        spark.stop()
        for pred in prediction_list:
            return models[int(pred.prediction)]
    