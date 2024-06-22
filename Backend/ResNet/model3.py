from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from functools import reduce
import seaborn as sns
import numpy as np
import itertools
import tensorflow as tf
import pathlib
from PIL import Image

# spark-submit --master local[*] --driver-memory 12g --executor-memory 12g --conf spark.ui.port=5050 model3.py


# ----------- Spark session and image dataset loading --------------

# Initialize a Spark session
print("Initializing Spark session...")
spark = SparkSession.builder.appName("CarImageClassification").getOrCreate()

# Load in dataset
print("Loading images from directory...")
alphard_df = spark.read.format("image").load("../toyota_cars/alphard").filter("NOT image.mode = 16").withColumn("label", lit(0))
iq_df = spark.read.format("image").load("../toyota_cars/iq").filter("mode != 16").withColumn("label", lit(1))

# Merge data frames
dataframes = [alphard_df, iq_df]

df = reduce(lambda first, second: first.union(second), dataframes)

# Repartition dataframe
df = df.repartition(200)

# Split dataframe into training and testing set
train, test = df.randomSplit([0.8, 0.2], 42)

# Print dataframe diagnostics
print(df.toPandas().size)
print(df.printSchema())


# --------------- Model building / training ---------------

# model: InceptionV3
# extracting feature from images
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features",
                                 modelName="InceptionV3")

# used as a multi class classifier
lr = LogisticRegression(maxIter=5, regParam=0.03,
                        elasticNetParam=0.5, labelCol="label")

# define a pipeline model
sparkdn = Pipeline(stages=[featurizer, lr])
spark_model = sparkdn.fit(train)

# not finished