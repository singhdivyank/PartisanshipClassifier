import seaborn as sns
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession

from consts import (
    SPARK_APP,
    PREDICTIONS_DF_PATH
)

def plot_hist(probabilities):
    """
    histogram for posterior probabilities using seaborn
    :param prob_df: Pandas Series representing posterior probabilities
    """
    probabilities = [prob for probs in probabilities for prob in probs]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(probabilities, bins=20, kde=True, color="blue", alpha=0.7, edgecolor="black")
    plt.title("Posterior Probabilities", fontsize=16)
    plt.xlabel("Probability", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    plt.savefig('hist.png')


if __name__ == '__main__':
    
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    predictions_df = spark.csv.read(PREDICTIONS_DF_PATH, header=True, inferSchema=True)
    prob_df = predictions_df.select('probability').toPandas()
    plot_hist(probabilities=prob_df['probability'])
