from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel

from consts import (
    SPARK_APP,
    DF_PATH,
    PREDICTIONS_DF_PATH,
    MODEL_PATH
)

def obtain_predictions(for_predictions):
    loaded_classifier = LogisticRegressionModel.load(MODEL_PATH)
    predictions_df = loaded_classifier.transform(for_predictions)
    # drop unwanted columns
    predictions_df = predictions_df.drop('rawPrediction')
    # save as CSV file
    predictions_df.toPandas().to_csv(PREDICTIONS_DF_PATH, header=True, index=False)


if __name__ == '__main__':
    
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    for_predictions = spark.read.csv(DF_PATH, header=True, inferSchema=True)
    obtain_predictions(for_predictions=for_predictions)