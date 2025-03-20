from pyspark.sql import SparkSession, functions as F

from consts import (
    SPARK_APP, 
    PREDICTIONS_DF_PATH, 
    TOP10_PATH, 
    METADATA_PATH,
    MISTAKES_PATH,
    MAXVOTE_PATH
)
from visualization import plot_hist, plot_values

def create_mistakes(predictions):
    metadata = spark.csv.read(METADATA_PATH, header=True, inferSchema=True)
    df = predictions.join(metadata, on='series', how='inner').drop('features', 'metadata', 'date')
    df = df.withColumn('matches', F.when(F.col('prediction')==F.col('issue_label'), True).otherwise(False))
    mistakes_df = df.filter(F.col('matches') == False).drop(F.col('matches'))
    mistakes_df.toPandas().to_csv(MISTAKES_PATH, header=True, index=True)

def perform_maxvote(predictions):
    voting_df = predictions.groupby('issue').agg(F.sum(F.when(F.col('prediction')==0, 1).otherwise(0)).alias('dem_count'), F.sum(F.when(F.col('prediction')==1, 1).otherwise(0)).alias('rep_count'))
    voting_df = voting_df.withColumn('majority_vote', F.when(F.col('dem_count')>F.col('rep_count'), 1).otherwise(0))
    voting_df.toPandas().to_csv(MAXVOTE_PATH, header=True, index=True)

def top_10_analysis():
    top_paras_df = spark.read.csv(TOP10_PATH, header=True, inferSchema=True)
    top10_series = top_paras_df.select("issue", "democrat_probability").toPandas()
    # plot histogram for partisan trends
    plot_hist(top10_series['democrat_probability'], fig_name='top10_hist.png', title="10 most partisan paragraphs")

def mistakes_analysis():
    # read mistakes csv
    mistakes_df = spark.read.csv(MISTAKES_PATH, header=True, inferSchema=True)
    # plot histogram for partisan trends
    plot_hist(probabilities=mistakes_df['democrat_probability'], fig_name='mistakes.png', title="Posterior Probabilities (Democratic) :: Mistakes")

def analyse_majority_vote():
    # read majority vote csv
    voting_df = spark.read.csv(MAXVOTE_PATH, header=True, inferSchema=True)
    voting_value_counts = voting_df.groupby('majority_vote').agg(total_dem_count=('dem_count', 'sum'), total_rep_count=('rep_count', 'sum')).reset_index()
    plot_values(values_df=voting_value_counts, index_col='majority_vote', cat1='total_dem_count', cat2='total_rep_count', cat1_label='Democrat Count', cat2_label='Republican Count', fig_name='max_vote_val.png')


if __name__ == '__main__':
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    predictions_df = spark.csv.read(PREDICTIONS_DF_PATH, header=True, inferSchema=True)
    predictions_df = predictions_df.drop('probability_array', 'rawPrediction', 'metadata', 'issue_label', 'words')
    # obtain the most partisan paragraphs and arrange in descending order
    predictions_df = predictions_df.filter(F.col('max_prob')>=0.7).orderBy(F.col('max_prob').desc())
    
    # obtain top10 most partisan paragraphs
    top10 = predictions_df.limit(10)
    top10.toPandas().to_csv(TOP10_PATH, header=True, index=False)
    # create mistakes dataframe
    create_mistakes(predictions=predictions_df)
    # perform majority voting
    perform_maxvote(predictions=predictions_df)

    # top-10 paragraph analysis
    top_10_analysis()
    # mistakes analysis
    mistakes_analysis()
    # majority vote analysis
    analyse_majority_vote()
