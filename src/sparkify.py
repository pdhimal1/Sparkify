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

from pyspark.ml.classification import GBTClassifier, LogisticRegression, LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import sum as Fsum, col
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.window import Window

DATA_FILE_MINI = "../data/mini_sparkify_event_data.json"
DATA_FILE_FULL = "../../../data/sparkify_event_data.json"
DATA_FILE = DATA_FILE_MINI


def init_spark():
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .appName("hw3") \
        .getOrCreate()
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
        .select('userId', (col('lifetime') / 3600 / 24).alias('lifetime'))

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

    # Playlist length
    playlist_length = data.select('userID', 'page') \
        .where(data.page == 'Add to Playlist') \
        .groupby('userID').count() \
        .withColumnRenamed('count', 'playlist_length')

    #  avg_songs_played per session
    avg_songs_played = data.where('page == "NextSong"') \
        .groupby(['userId', 'sessionId']) \
        .count() \
        .groupby(['userId']) \
        .agg({'count': 'avg'}) \
        .withColumnRenamed('avg(count)', 'avg_songs_played')

    # artist count
    artist_count = data \
        .filter(data.page == "NextSong") \
        .select("userId", "artist") \
        .dropDuplicates() \
        .groupby("userId") \
        .count() \
        .withColumnRenamed("count", "artist_count")

    # number of sessions
    num_sessions = data \
        .select("userId", "sessionId") \
        .dropDuplicates() \
        .groupby("userId") \
        .count() \
        .withColumnRenamed('count', 'num_sessions')

    cols = ["lifetime",
            "total_songs",
            "num_thumb_up",
            'num_thumb_down',
            'add_friend',
            'playlist_length',
            'avg_songs_played',
            'artist_count',
            'num_sessions'
            ]

    return cols, [time_since_registration,
                  total_songs_listened,
                  thumbs_up,
                  thumbs_down,
                  referring_friends,
                  playlist_length,
                  avg_songs_played,
                  artist_count,
                  num_sessions]


def create_cross_validator_GBT(folds=3):
    # initialize classifier
    GradBoostTree = GBTClassifier()

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    depth = [3, 5, 7]  # number of features
    iterations = [10, 20, 30]  # default is 20
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


def get_model_GBT(crossValidation=False, folds=3):
    if crossValidation:
        cross_validator = create_cross_validator_GBT(folds=folds)
        return cross_validator
    else:
        return GBTClassifier()


def create_cross_validator_LGR(folds=3):
    lgr = LogisticRegression()

    # We use a ParamGridBuilder to construct a grid of parameters to search over
    iterations = [90, 100, 110]  # default is 100
    regParam = [0.1, 0.2]
    elasticNetParam = [0.8, 0.9]

    paramGrid = ParamGridBuilder() \
        .addGrid(lgr.maxIter, iterations) \
        .addGrid(lgr.elasticNetParam, elasticNetParam) \
        .addGrid(lgr.regParam, regParam) \
        .build()

    evaluator = MulticlassClassificationEvaluator(metricName='f1')

    cross_validator = CrossValidator(estimator=lgr,
                                     estimatorParamMaps=paramGrid,
                                     evaluator=evaluator,
                                     numFolds=folds)
    return cross_validator


def get_model_LGR(crossValidation=False, folds=3):
    if crossValidation:
        cross_validator = create_cross_validator_LGR(folds=folds)
        return cross_validator
    else:
        return LogisticRegression()

    # predictions = lr.transform(test)
    # evaluator = BinaryClassificationEvaluator(labelCol='churned_num')
    #
    # print('Logistic regresstion, test set, Area Under ROC', evaluator.evaluate(predictions))


def create_cross_validator_SVC(folds=3):
    svc = LinearSVC()

    # We use a ParamGridBuilder to construct a grid of parameters to search over
    iterations = [90, 100, 110]  # default is 100
    regParam = [0.1, 0.2, 0.3]

    paramGrid = ParamGridBuilder() \
        .addGrid(svc.maxIter, iterations) \
        .addGrid(svc.regParam, regParam) \
        .build()

    evaluator = MulticlassClassificationEvaluator(metricName='f1')

    cross_validator = CrossValidator(estimator=svc,
                                     estimatorParamMaps=paramGrid,
                                     evaluator=evaluator,
                                     numFolds=folds)
    return cross_validator


def get_model_SVC(crossValidation=False, folds=3):
    if crossValidation:
        cross_validator = create_cross_validator_SVC(folds)
        return cross_validator
    else:
        return LinearSVC()


def create_cross_validator_RandomForest(folds=3):
    rf_classifier = RandomForestClassifier()
    # We use a ParamGridBuilder to construct a grid of parameters to search over
    maxDepth = [4, 5, 6]  # default is 5
    paramGrid = ParamGridBuilder() \
        .addGrid(rf_classifier.maxDepth, maxDepth) \
        .build()
    evaluator = MulticlassClassificationEvaluator(metricName='f1')
    cross_validator = CrossValidator(estimator=rf_classifier,
                                     estimatorParamMaps=paramGrid,
                                     evaluator=evaluator,
                                     numFolds=folds)
    return cross_validator


def get_model_RandomForest(crossValidation=False, folds=3):
    if crossValidation:
        cross_validator = create_cross_validator_RandomForest(folds)
        return cross_validator
    else:
        return RandomForestClassifier()


def hybrid_function(prediction_GBT, prediction_LGR, prediction_SVM, prediction_rf):
    sum_predictions = prediction_GBT + prediction_LGR + prediction_SVM + prediction_rf
    if sum_predictions >= 1:
        return 1.0
    else:
        return 0.0


def main(
        outFile,
        timeStamp,
        crossValidation=False,
        folds=3):
    time_start = time.time()
    print("Using data from ", DATA_FILE)
    print("Using data from ", DATA_FILE, file=outFile)
    spark = init_spark()
    data = read_data(spark)

    # userId, registration, ts, song, page, sessionId
    data = data.drop(*['firstName', 'lastName', 'id_copy'])
    data = data.filter(data["userId"] != "").filter(data['registration'].isNotNull())

    # milliseconds to microseconds
    time_unit_udf = F.udf(lambda x: x / 1000, DoubleType())
    data = data.withColumn("registration", time_unit_udf("registration"))
    data = data.withColumn("ts", time_unit_udf("ts"))
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

    total_churners = data.filter(data['churn'] == 1).count()
    print("The total number of churning instances: ", total_churners)
    print("The total number of churners instances: ", total_churners, file=outFile)

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
    # data.show(vertical=True, n=2)
    data = data.select('userID', 'unScaled_features', 'label')
    # data.show(vertical=True, n=2)

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

    ### RUNNING GRADIENT BOOSTED TREES ###
    if crossValidation:
        print("Running Gradient Boosted Trees cross validation with {} folds ...".format(folds))
    else:
        print("Running Gradient Boosted Trees without cross validation ...")
    model = get_model_GBT(crossValidation, folds)
    cvModel_GradBoostTree = model.fit(trainingDF)
    # cvModel_GradBoostTree.avgMetrics
    results_GradBoostTree = cvModel_GradBoostTree.transform(testDF)
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
    print('Accuracy: {:.2f}'.format(accuracy))
    print('Accuracy: {:.2f}'.format(accuracy), file=outFile)
    print('F1 Score: {:.2f}'.format(f1Score))
    print('F1 Score: {:.2f}'.format(f1Score), file=outFile)

    '''
    print("Feature importance: ")
    print("Feature importance: ", file=outFile)
    for feat, col in zip(cvModel_GradBoostTree.featureImportances, column_names):
        print("\t", col, ": ", feat)
        print("\t", col, ": ", feat, file=outFile)
    '''

    ### RUNNING LOGISTIC REGRESSION MODEL ###
    if crossValidation:
        print("Running Logistic Regression cross validation with {} folds ...".format(folds))
    else:
        print("Running Logistic Regression without cross validation ...")

    model = get_model_LGR(crossValidation, folds)
    cvModel_lgr = model.fit(trainingDF)

    results_lgr = cvModel_lgr.transform(testDF)
    results_lgr = results_lgr.select('userID', 'label', 'prediction')
    results_lgr.show(10)

    if crossValidation:
        print("Best model selected from cross validation:\n", cvModel_lgr.bestModel)
        print("Best model selected from cross validation:\n", cvModel_lgr.bestModel, file=outFile)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(results_lgr, {evaluator.metricName: "accuracy"})
    f1Score = evaluator.evaluate(results_lgr, {evaluator.metricName: "f1"})
    print('Logistic Regression Metrics:')
    print('Logistic Regression Metrics:', file=outFile)
    print('Accuracy: {:.2f}'.format(accuracy))
    print('Accuracy: {:.2f}'.format(accuracy), file=outFile)
    print('F1 Score: {:.2f}'.format(f1Score))
    print('F1 Score: {:.2f}'.format(f1Score), file=outFile)

    ### Running SVM Model ###
    if crossValidation:
        print("Running SVM cross validation with {} folds ...".format(folds))
    else:
        print("Running SVM without cross validation ...")

    model = get_model_SVC(crossValidation, folds)
    cvModel_svc = model.fit(trainingDF)

    results_svc = cvModel_svc.transform(testDF)
    results_svc = results_svc.select('userID', 'label', 'prediction')
    results_svc.show(10)

    if crossValidation:
        print("Best model selected form cross validation:\n", cvModel_svc.bestModel)
        print("Best model selected form cross validation:\n", cvModel_svc.bestModel, file=outFile)

    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(results_svc, {evaluator.metricName: "accuracy"})
    f1Score = evaluator.evaluate(results_svc, {evaluator.metricName: "f1"})
    print('SVM Metrics:')
    print('SVM Metrics:', file=outFile)
    print('Accuracy: {:.2f}'.format(accuracy))
    print('Accuracy: {:.2f}'.format(accuracy), file=outFile)
    print('F1 Score: {:.2f}'.format(f1Score))
    print('F1 Score: {:.2f}'.format(f1Score), file=outFile)

    model = get_model_RandomForest()
    cvModel_rf = model.fit(trainingDF)

    # Make Predictions
    results_rf = cvModel_rf.transform(testDF).select('userID', 'label', 'prediction')
    results_rf.show(10)

    if crossValidation:
        print("Best model selected form cross validation:\n", results_rf.bestModel)
        print("Best model selected form cross validation:\n", results_rf.bestModel, file=outFile)

    # Get Results
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(results_rf, {evaluator.metricName: "accuracy"})
    f1Score = evaluator.evaluate(results_rf, {evaluator.metricName: "f1"})
    print('Random Forest Metrics:')
    print('Random Forest Metrics:', file=outFile)
    print('Accuracy: {:.2f}'.format(accuracy))
    print('Accuracy: {:.2f}'.format(accuracy), file=outFile)
    print('F1 Score: {:.2f}'.format(f1Score))
    print('Accuracy: {:.2f}'.format(accuracy), file=outFile)

    results_GradBoostTree = results_GradBoostTree.withColumnRenamed("prediction", "prediction_GBT")
    results_lgr = results_lgr.select('userId', 'prediction').withColumnRenamed("prediction", "prediction_LGR")
    results_svc = results_svc.select('userId', 'prediction').withColumnRenamed("prediction", "prediction_SVC")
    results_rf = results_rf.select('userId', 'prediction').withColumnRenamed("prediction", "prediction_RF")
    results = results_GradBoostTree \
        .join(results_lgr, 'userID', 'outer') \
        .join(results_svc, 'userID', 'outer') \
        .join(results_rf, 'userID', 'outer')

    udf_hybrid_calc_function = F.udf(hybrid_function, DoubleType())
    results = results.withColumn("prediction",
                                 udf_hybrid_calc_function(
                                     "prediction_GBT",
                                     "prediction_SVC",
                                     "prediction_LGR",
                                     "prediction_RF"))
    results.show()
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    accuracy = evaluator.evaluate(results, {evaluator.metricName: "accuracy"})
    f1Score = evaluator.evaluate(results, {evaluator.metricName: "f1"})
    print('Hybrid Metrics:')
    print('Hybrid Metrics:', file=outFile)
    print('Accuracy: {:.2f}'.format(accuracy))
    print('Accuracy: {:.2f}'.format(accuracy), file=outFile)
    print('F1 Score: {:.2f}'.format(f1Score))
    print('F1 Score: {:.2f}'.format(f1Score), file=outFile)

    print(results.count())
    if results.count() > 1000:
        file_name = '../out/predictions-' + timeStamp + '.csv'
        results.write.option("header", "true").csv(file_name)

    time_end = time.time()
    print("Total time to run Script: {} minutes".format((time_end - time_start) / 60))
    print("Total time to run Script: {} seconds".format(time_end - time_start), file=outFile)
    print("Total time to run Script: {} minutes".format((time_end - time_start) / 60), file=outFile)

    spark.stop()


if __name__ == "__main__":
    # cross validation
    crossValidation = False
    folds = 3

    '''
    Best model selected from cross validation:
    GBTClassificationModel: uid = GBTClassifier_ecaf70bbfc1e, numTrees=20, numClasses=2, numFeatures=5
    
    Best model selected from cross validation:
    LogisticRegressionModel: uid=LogisticRegression_8d843b9072b8, numClasses=2, numFeatures=5
    
    Best model selected form cross validation:
    LinearSVCModel: uid=LinearSVC_b8cfaa1e744d, numClasses=2, numFeatures=5
    '''
    # to use the full dataset, set this to DATA_FILE_FULL, and make sure DATA_FILE_FULL exists
    DATA_FILE = DATA_FILE_MINI

    time_stamp = str(int(time.time()))
    out_file_name = '../out/output-' + time_stamp + '.txt'
    out_file = open(out_file_name, 'w')
    main(
        out_file,
        time_stamp,
        crossValidation=crossValidation,
        folds=folds)
    print("Check ", out_file.name)
