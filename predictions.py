from pyspark.sql import SparkSession, functions as F
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

from consts import (
    SPARK_APP,
    TRAIN_DF_PATH,
    TEST_DF_PATH,
    DF_PATH,
    PREDICTIONS_DF_PATH,
    MODEL_PATH
)

LR = LogisticRegression(featuresCol = 'features', labelCol = 'issue_label', family = 'multinomial')
EVALUATOR = MulticlassClassificationEvaluator(labelCol='issue_label', predictionCol='prediction')

def modify_predictions(predictions):
    # add additional columns for democrat and republican probabilities
    predictions = predictions.withColumn('probability_array', vector_to_array(predictions['probability'])).withColumn('max_prob', F.array_max('probability_array')).withColumn('democrat_probability', F.col("probability_array").getItem(0)).withColumn('republican_probability', F.col("probability_array").getItem(1))
    predictions = predictions.drop(F.col('probability_array')).orderBy(F.desc('max_prob')).filter(F.col('prediction') == F.col('issue_label'))
    return predictions

def train_classifier():
    # load train and test datasets
    train, test = spark.read.csv(TRAIN_DF_PATH, header=True, inferSchema=True), \
        spark.read.csv(TEST_DF_PATH, header=True, inferSchema=True)
    # train Logistic Regression classifier
    model = LR.fit(train)
    # save model
    model.write().overwrite().save(MODEL_PATH)
    # make prediction
    predictions = model.transform(test).drop('rawPrediction')
    # evaluate performance
    accuracy, f1 = EVALUATOR.evaluate(predictions, {EVALUATOR.metricName: "accuracy"}), \
        EVALUATOR.evaluate(predictions, {EVALUATOR.metricName: "f1"})
    # modify predictions dataframe
    predictions_df = modify_predictions(predictions=predictions)
    # save as CSV file
    predictions_df.toPandas().to_csv(PREDICTIONS_DF_PATH, header=True, index=False)
    print(f"accuracy: {accuracy}\nf1 score: {f1}")

def obtain_predictions(for_predictions):
    loaded_classifier = LogisticRegressionModel.load(MODEL_PATH)
    predictions_df = loaded_classifier.transform(for_predictions)
    # drop unwanted columns
    predictions_df = predictions_df.drop('rawPrediction')
    # modify predictions dataframe
    predictions = modify_predictions(predictions=predictions)
    # save as CSV file
    predictions_df.toPandas().to_csv(PREDICTIONS_DF_PATH, header=True, index=False)


if __name__ == '__main__':
    
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    for_predictions = spark.read.csv(DF_PATH, header=True, inferSchema=True)
    # training the model
    train_classifier()
    obtain_predictions(for_predictions=for_predictions)
