from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# PREPARE AND TRAIN - VERSION 1 #
# ============================= #
def prepare_data(x_train, y_train, features, target):
    x_train = x_train.withColumn('row_index', F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
    y_train = y_train.withColumn('row_index', F.row_number().over(Window.orderBy(F.monotonically_increasing_id())))
    train = x_train.join(y_train, on=["row_index"]).drop("row_index")
    # features
    f_columns = ",".join(features).split(",")
    # target
    f_target = ",".join(target).split(",")
    f_target = list(map(lambda c: F.col(c).alias("label"), f_target))
    # all columns
    all_columns = ",".join(features + target).split(",")
    all_columns = list(map(lambda c: F.col(c), all_columns))
    # model data set
    train_data = train.select(all_columns)
    assembler = VectorAssembler(inputCols=f_columns, outputCol='features')
    train_data = assembler.transform(train_data)
    train_data = train_data.select('features', f_target[0])
    return train_data


def train_and_predict(x_train, y_train, features, target, evaluator):
    train_data = prepare_data(x_train, y_train, features, target)
    lr = LogisticRegression(maxIter=10, regParam=0.01)
    # train model
    train, test = train_data.randomSplit([0.8, 0.2], seed=555)
    model = lr.fit(train)
    # check the model on the test data
    prediction = model.transform(test)
    prediction.show(5)
    evaluation_result = evaluator.evaluate(prediction)
    print("Evaluation result: {}".format(evaluation_result))
    return model
