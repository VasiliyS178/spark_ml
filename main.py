# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.utils import prepare_data, train_and_predict

spark = SparkSession.builder.appName("mllib_prepare_and_train_app").master("local[*]").getOrCreate()

train_path = "ml_prj/x_train.csv"
target_path = "ml_prj/y_train.csv"
model_dir = "ml_prj/models"

# model evaluator
evaluator = BinaryClassificationEvaluator() \
        .setMetricName("areaUnderROC") \
        .setLabelCol("label") \
        .setPredictionCol("prediction")

# read data from hdfs
x_train = spark\
    .read\
    .format("csv")\
    .options(inferSchema=True, header=True, sep=";")\
    .load(train_path)
x_train.show(3)
print(x_train.schema.json())

# target
y_train = spark\
    .read\
    .format("csv")\
    .options(inferSchema=True, header=True, sep=";")\
    .load(target_path)
y_train.show(3)

features = ["maxPlayerLevel,numberOfAttemptedLevels,attemptsOnTheHighestLevel,totalNumOfAttempts,\
averageNumOfTurnsPerCompletedLevel,doReturnOnLowerLevels,numberOfBoostersUsed,fractionOfUsefullBoosters,\
totalScore,totalBonusScore,totalStarsCount,numberOfDaysActuallyPlayed"]

target = ["label"]

train_data = prepare_data(x_train, y_train, features, target)
train_data.show(3)

model = train_and_predict(x_train, y_train, features, target)
model.write().overwrite().save(model_dir)