from pyspark.sql import SparkSession, functions as F, Window

from visualization import plot_hist
from consts import (
    SPARK_APP,
    PREDICTIONS_DF_PATH,
    DEM_PATH,
    REP_PATH,
    HIST_FOLDER
)

def series_hist(probability_series):
    """
    histogram for posterior series but for each series
    :param probabilities: Pandas Series representing posterior probabilities
    """

    unique_series = probability_series["series"].unique()
    for idx, series in enumerate(unique_series):
        prob_df = probability_series[probability_series['series'] == series]
        plot_hist(probabilities=prob_df['democrat_probabability'], fig_name=f"{HIST_FOLDER}/hist_{idx}.png", title=f"Series:{series} Posterior Probabilities (Democratic)")

def identify_series(series_df):
    """
    identify the democratic and republican series
    """

    dem_idx, rep_idx = [], []
    # obtain all unique series
    unique_series = series_df['series'].unique()
    # iterate over each series
    for _, series in enumerate(unique_series):
        # obtain democrat probability
        dem_prob = series_df[series_df['series'] == series]['democrat_probability']
        # if mean>0.5: democratic else republican
        if dem_prob.mean()>0.5:
            dem_idx.append(series)
        else:
            rep_idx.append(series)
    
    return dem_idx, rep_idx

def rerank(df, col_name):
    """
    rerank paragraghs of each series based
    on Mean Average Precision    
    """

    series_ap = (df.groupBy('series').agg(F.avg(col_name).alias('AveragePrecision')))
    window_spec = Window.partitionBy('series').orderBy(F.desc(col_name))
    ranked_predictions = df.withColumn('rank', F.row_number().over(window_spec))
    ranked_predictions = ranked_predictions.join(series_ap, 'series').withColumn('score', F.when(F.col('rank')==1.0, 1.0).otherwise(0))
    ranked_predictions = ranked_predictions.filter(F.col('score')==1.0).drop('rank', 'AveragePrecision', 'score', col_name)
    return ranked_predictions

def series_analysis(predictions):
    para_df = predictions.select('series', 'issue', 'date', 'cleaned_paragraph', 'democrat_probability', 'republican_probability')
    dem_series = para_df.select('series', 'democrat_probability').toPandas()
    dem_idx, rep_idx = identify_series(series_df=dem_series)
    dem_df, rep_df = para_df.filter(F.col('series').isin(dem_idx)), \
        para_df.filter(F.col('series').isin(rep_idx)).drop('democrat_probability')
    dem_df, rep_df = rerank(df=dem_df, col_name='democrat_probability'), \
        rerank(df=rep_df, col_name='republican_probability')
    dem_df.toPandas().to_csv(DEM_PATH, header=True, index=False)
    rep_df.toPandas().to_csv(REP_PATH, header=True, index=False)
    series_hist(dem_series)


if __name__ == '__main__':
    
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    predictions_df = spark.csv.read(PREDICTIONS_DF_PATH, header=True, inferSchema=True)
    series_analysis(predictions=predictions_df)
