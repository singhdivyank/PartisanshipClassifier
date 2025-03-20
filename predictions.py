import numpy as np

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

from visualization import CreatePlots
from consts import (
    SPARK_APP,
    TRAIN_DF_PATH,
    DF_PATH,
    PREDICTIONS_DF_PATH,
    MODEL_PATH
)

LR = LogisticRegression(featuresCol = 'features', labelCol = 'issue_label', family = 'multinomial')
MULTI_CLASS_EVALUATOR = MulticlassClassificationEvaluator(labelCol='issue_label', predictionCol='prediction')
BIN_CLASS_EVALUATOR = BinaryClassificationEvaluator(labelCol='issue_label', rawPredictionCol='rawPrediction')

def vector_to_array(vector):
    return vector.toArray().tolist()

def train_classifier():
    # load train dataset
    train_df = spark.read.csv(TRAIN_DF_PATH, header=True, inferSchema=True)
    
    try:
        # train Logistic Regression classifier
        model = LR.fit(train_df)
        # save model
        model.write().overwrite().save(MODEL_PATH)
        print(f"saved model weights, location: {MODEL_PATH}")
    except Exception as error:
        print(f"error saving model :: {str(error)}")
        return 

def evaluation():

    evaluation_results = {
        "auc-roc": 0.0,
        "auc-pr": 0.0,
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1-score": 0.0
    }

    predictions = spark.read.csv(PREDICTIONS_DF_PATH, header=True, inferSchema=True)
    try:
        # compute auc-roc and auc-pr
        roc = BIN_CLASS_EVALUATOR.evaluate(predictions, {BIN_CLASS_EVALUATOR.metricName: "areaUnderROC"})
        pr = BIN_CLASS_EVALUATOR.evaluate(predictions, {BIN_CLASS_EVALUATOR.metricName: "areaUnderPR"})
        # compute accuracy, precision, recall, f1
        accuracy = MULTI_CLASS_EVALUATOR.evaluate(predictions, {MULTI_CLASS_EVALUATOR.metricName: "accuracy"})
        f1 = MULTI_CLASS_EVALUATOR.evaluate(predictions, {MULTI_CLASS_EVALUATOR.metricName: "f1"})
        precision = MULTI_CLASS_EVALUATOR.evaluate(predictions, {MULTI_CLASS_EVALUATOR.metricName: "weightedPrecision"})
        recall = MULTI_CLASS_EVALUATOR.evaluate(predictions, {MULTI_CLASS_EVALUATOR.metricName: "weightedRecall"})
        evaluation_results.update({
            "auc-roc": roc,
            "auc-pr": pr,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1
        })
    except Exception as error:
        print(f"Error while evaluating classifier :: {str(error)}")
    
    return evaluation_results

def obtain_predictions(for_predictions):
    loaded_classifier = LogisticRegressionModel.load(MODEL_PATH)
    predictions_df = loaded_classifier.transform(for_predictions)
    # add additional columns for democrat and republican probabilities
    predictions = predictions_df.withColumn('probability_array', vector_to_array(predictions['probability'])).withColumn('max_prob', F.array_max('probability_array')).withColumn('democrat_probability', F.col("probability_array").getItem(0)).withColumn('republican_probability', F.col("probability_array").getItem(1)).drop(F.col('probability_array'))
    # save as CSV file
    predictions.toPandas().to_csv(PREDICTIONS_DF_PATH, header=True, index=False)
    return predictions

def obtain_eval_labels(predictions):
    
    # y_true
    y_true = final_predictions.select('issue_label').rdd.flatMap(lambda x: x).collect()
    # y_preds
    y_preds = final_predictions.select('prediction').rdd.flatMap(lambda x: x).collect()
    # y_scores
    vector_to_array_udf = F.udf(vector_to_array, T.ArrayType(T.DoubleType()))
    preds = predictions.withColumn("probability_array", vector_to_array_udf(predictions["probability"]))
    score_df = preds.select('probability_array').toPandas()
    y_scores = np.array(score_df['probability_array'].tolist())
    
    return y_true, y_preds, y_scores


if __name__ == '__main__':
    
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    for_predictions = spark.read.csv(DF_PATH, header=True, inferSchema=True)
    # train classifier
    train_classifier()
    # make predictions
    final_predictions = obtain_predictions(for_predictions=for_predictions)
    # evaluate trained classifier
    evaluation_results = evaluation()
    # obtain y_true, y_preds, y_scores
    y_true, y_preds, y_scores = obtain_eval_labels(predictions=final_predictions)
    plots = CreatePlots(y_true=y_true, y_preds=y_preds,y_scores=y_scores, num_classes=y_scores.shape[1])
    plots.create_plots()
    