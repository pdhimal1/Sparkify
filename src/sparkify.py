'''
Prakash Dhimal
Manav Garkel
George Mason University
CS 657 Mining Massive Datasets
Final Project: Sparkify

Data from:

https://udacity-dsnd.s3.amazonaws.com/sparkify/sparkify_event_data.json
https://udacity-dsnd.s3.amazonaws.com/sparkify/mini_sparkify_event_data.json
'''
import time

from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import sum as Fsum, col
from pyspark.sql.types import IntegerType
from pyspark.sql.window import Window

DATA_FILE_MINI = "../data/mini_sparkify_event_data.json"
DATA_FILE_FULL = "../../../data/sparkify_event_data.json"
DATA_FILE = DATA_FILE_MINI


def init_spark():
    spark = SparkSession.builder.master("local[*]").appName("sparkify").getOrCreate()
    return spark


def read_data(spark):
    data = spark.read.json(DATA_FILE)
    return data


def add_churn_column(data):
    # Define a flag function
    flag_cancelation_event = F.udf(lambda x: 1 if x == "Cancellation Confirmation" else 0, IntegerType())
    # apply to the dataframe
    data = data.withColumn("churn", flag_cancelation_event("page"))

    # Define window bounds
    windowval = Window.partitionBy("userId") \
        .rangeBetween(Window.unboundedPreceding,
                      Window.unboundedFollowing)
    # Applying the window
    data = data.withColumn("churn", Fsum("churn").over(windowval))

    return data


def get_label_per_user(data):
    # label, comes from the churn
    label = data \
        .select('userId', col('churn').alias('label')) \
        .dropDuplicates()
    return label


def add_features(data):
    # time since registration
    time_since_registration = data \
        .select('userId', 'registration', 'ts') \
        .withColumn('lifetime', (data.ts - data.registration)) \
        .groupBy('userId') \
        .agg({'lifetime': 'max'}) \
        .withColumnRenamed('max(lifetime)', 'lifetime') \
        .select('userId', (col('lifetime') / 1000 / 3600 / 24).alias('lifetime'))

    # total songs listened
    total_songs_listened = data \
        .select('userID', 'song') \
        .groupBy('userID') \
        .count() \
        .withColumnRenamed('count', 'total_songs')

    # thumbs up
    thumbs_up = data \
        .select('userID', 'page') \
        .where(data.page == 'Thumbs Up') \
        .groupBy('userID') \
        .count() \
        .withColumnRenamed('count', 'num_thumb_up')

    # thumbs down
    thumbs_down = data \
        .select('userID', 'page') \
        .where(data.page == 'Thumbs Down') \
        .groupBy('userID') \
        .count() \
        .withColumnRenamed('count', 'num_thumb_down')

    # referring friends
    referring_friends = data \
        .select('userID', 'page') \
        .where(data.page == 'Add Friend') \
        .groupBy('userID') \
        .count() \
        .withColumnRenamed('count', 'add_friend')

    cols = ["lifetime",
            "total_songs",
            "num_thumb_up",
            'num_thumb_down',
            'add_friend']

    return cols, [time_since_registration,
                  total_songs_listened,
                  thumbs_up,
                  thumbs_down,
                  referring_friends]


def create_cross_validator(folds=3):
    # initialize classifier
    GradBoostTree = GBTClassifier()

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    depth = [3, 5, 7]  # number of features
    iterations = [10, 20, 30]
    bins = [16, 32, 64]
    #
    # maxDepth=5, maxBins=32, maxIter=20
    param_grid = ParamGridBuilder() \
        .addGrid(GradBoostTree.maxDepth, depth) \
        .addGrid(GradBoostTree.maxIter, iterations) \
        .build()

    evaluator = MulticlassClassificationEvaluator(metricName='f1')

    cross_validator = CrossValidator(estimator=GradBoostTree,
                                     estimatorParamMaps=param_grid,
                                     evaluator=evaluator,
                                     numFolds=folds)
    return cross_validator


def get_model(maxIter, crossValidation=False, folds=3):
    if crossValidation:
        cross_validator = create_cross_validator(folds=folds)
        return cross_validator
    else:
        return GBTClassifier(maxIter=maxIter, seed=42)


def logistic_regression(train, test):
    lr = LogisticRegression(featuresCol="scaled_features", labelCol="churned_num")
    lr = lr.fit(train)

    predictions = lr.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol='churned_num')

    print('Logistic regresstion, test set, Area Under ROC', evaluator.evaluate(predictions))


def main(
        outFile,
        timeStamp,
        crossValidation=False,
        maxIter=5,
        folds=3):
    time_start = time.time()
    print("Using data from ", DATA_FILE)
    print("Using data from ", DATA_FILE, file=outFile)
    spark = init_spark()
    data = read_data(spark)
    '''
    Adding churn column to each instance
    '''
    data = add_churn_column(data)
    # to show the different pages that the users are looking at
    pages = data.select("page").dropDuplicates()
    pages.show()
    print(pages.toPandas(), file=outFile)
    print(data.limit(10).toPandas(), file=outFile)

    unique_userIDs = data.select("userID").dropDuplicates().count()
    print("The total number of users:", unique_userIDs)
    print("The total number of users:", unique_userIDs, file=outFile)

    # todo - look at this
    total_churners = data.agg(Fsum("churn")).collect()[0][0]
    print("The total number of churners are:", total_churners)
    print("The total number of churners are:", total_churners, file=outFile)

    '''
    Each users are assigned a label
    
    '''
    label_per_user = get_label_per_user(data)
    column_names, features = add_features(data)

    data = features.pop()
    while len(features) > 0:
        data = data.join(features.pop(), 'userID', 'outer')

    data = data.join(label_per_user, 'userID', 'outer').fillna(0)
    data.show(vertical=True, n=2)

    # Vector assembler
    assembler = VectorAssembler(inputCols=column_names, outputCol="unScaled_features")
    data = assembler.transform(data)

    # let go of all the feature columns
    data.show(vertical=True, n=2)
    data = data.select('userID', 'unScaled_features', 'label')
    data.show(vertical=True, n=2)

    # scale the features
    scaler = StandardScaler(inputCol="unScaled_features", outputCol="features", withStd=True)
    scalerModel = scaler.fit(data)
    data = scalerModel.transform(data)

    print("Data transformation is complete ...")
    print("Number of users in the data: ", data.count())
    print("Number of users in the data: ", data.count(), file=outFile)

    # train test split
    trainTest = data.randomSplit([0.8, 0.2])
    trainingDF = trainTest[0]
    testDF = trainTest[1]

    if crossValidation:
        print("Running Gradient Boosted Trees cross validation with {} folds ...".format(folds))
    else:
        print("Running Gradient Boosted Trees without cross validation ...")
    model = get_model(maxIter, crossValidation, folds)
    cvModel_GradBoostTree = model.fit(trainingDF)
    # cvModel_GradBoostTree.avgMetrics
    results_GradBoostTree = cvModel_GradBoostTree.transform(testDF)
    # todo -    rawPrediction|         probability  ?
    results_GradBoostTree = results_GradBoostTree.select('userID', 'label', 'prediction')
    results_GradBoostTree.show(10)

    if crossValidation:
        print("Best model selected from cross validation:\n", cvModel_GradBoostTree.bestModel)
        print("Best model selected from cross validation:\n", cvModel_GradBoostTree.bestModel, file=outFile)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(results_GradBoostTree, {evaluator.metricName: "accuracy"})
    f1Score = evaluator.evaluate(results_GradBoostTree, {evaluator.metricName: "f1"})
    print('Gradient Boosted Trees Metrics:')
    print('Gradient Boosted Trees Metrics:', file=outFile)
    print('Accuracy: {}'.format(accuracy))
    print('Accuracy: {}'.format(accuracy), file=outFile)
    print('F1 Score:{}'.format(f1Score))
    print('F1 Score:{}'.format(f1Score), file=outFile)

    time_end = time.time()
    print("Total time to run Script: {} minutes".format((time_end - time_start) / 60))
    print("Total time to run Script: {} seconds".format(time_end - time_start), file=outFile)
    print("Total time to run Script: {} minutes".format((time_end - time_start) / 60), file=outFile)

    print(results_GradBoostTree.count())
    if results_GradBoostTree.count() > 1000:
        file_name = '../out/predictions-' + timeStamp + '.csv'
        results_GradBoostTree.write.option("header", "true").csv(file_name)
    spark.stop()


if __name__ == "__main__":
    # cross validation
    crossValidation = False
    folds = 3

    # gradient boost trees
    # Best model selected from cross validation:
    # GBTClassificationModel: uid = GBTClassifier_ecaf70bbfc1e, numTrees=20, numClasses=2, numFeatures=5
    maxIter = 5

    # to use the full dataset
    DATA_FILE = DATA_FILE_FULL

    time_stamp = str(int(time.time()))
    out_file_name = '../out/output-' + time_stamp + '.txt'
    out_file = open(out_file_name, 'w')
    main(
        out_file,
        time_stamp,
        crossValidation=crossValidation,
        folds=folds,
        maxIter=maxIter)
    print("Check ", out_file.name)
