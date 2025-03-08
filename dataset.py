import re

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer

from visualization import value_plots
from consts import (
    NLTK_WORDS, 
    CUSTOM_WORDS,
    PARQUET_PATH,
    LABELS_CSV,
    TRAIN_DF_PATH,
    TEST_DF_PATH,
    SPARK_APP,
    DF_PATH,
)

STOPWORDS = list(set(NLTK_WORDS+CUSTOM_WORDS))

def labels_df(csv_df):
    # drop na from series
    csv_df = csv_df.dropna(subset='series')
    # assign labels
    csv_df = csv_df.withColumn('label', F.when(F.array_contains(F.split(F.col('contents'), '; '), 'democratic'), 0.0).when(F.array_contains(F.split(F.col('contents'), '; '), 'republican'), 1.0).when(F.array_contains(F.split(F.col('contents'), '; '), 'independent'), 2.0).otherwise(3.0))
    csv_df = csv_df.select('series', F.col('contents').alias('metadata'), F.col('label').alias('issue_label'))
    # remove xref from series
    csv_df = csv_df.filter(~F.col('metadata').isin('xref', '(no report.)'))
    return csv_df

def clean_para(paragraph):
    sent = ''

    for word in paragraph.split(','):
        word = word.strip()
        if word:
            sent += word + ", "
    
    return sent

def feature_extraction(df, inputCol): 
    tokenize_data =  RegexTokenizer(inputCol=inputCol, outputCol="words", minTokenLength=1, pattern="\\W").transform(df)
    hash_data = HashingTF(inputCol="words", outputCol="rawFeatures").transform(tokenize_data)
    idfModel = IDF(inputCol="rawFeatures", outputCol="features").fit(hash_data)
    return idfModel.transform(hash_data)

def preprocess(content):
    if not content.strip():
        return content
    
    # Remove non-alphabetical characters except commas, spaces, and \n\n
    content = re.sub(r'[^\w,\s\n]', ' ', content)
    # Remove extra spaces around commas
    content = re.sub(r'\s*,\s*', ',', content).strip()
    # Remove extra spaces between words
    content = re.sub(r'\s+', ' ', content).strip()
    # remove words upto 3 characters
    content = re.sub(r'\b\w{1,3}\b', '', content).strip()
    # filter out stopwords
    filtered_words = [word for word in content.split() if word not in STOPWORDS]
    # Remove empty strings from the list
    filtered_words[:] = [word.strip() for word in filtered_words if word]
    return ' '.join(word.lower() for word in filtered_words)

def perform_split():
    to_split_df = spark.read.csv(DF_PATH, header=True, inferSchema=True)
    series_df = to_split_df.select('series').distinct()
    series_train, series_test = series_df.randomSplit([0.8, 0.2], seed=0)
    train_df, test_df = to_split_df.join(series_train, on='series', how='inner'), to_split_df.join(series_test, on='series', how='inner')
    # train on only democrat and republican
    train_df = train_df.filter(F.col('issue_label').isin(0, 1))
    # visualise value counts
    value_plots(values_df=train_df.groupby('issue_label').count().toPandas(), fig_name='train_value_counts.png')
    value_plots(values_df=test_df.groupby('issue_label').count().toPandas(), fig_name='test_value_counts.png')
    # save as pandaas dataframe
    train_df.toPandas().to_csv(TRAIN_DF_PATH, header=True, index=False)
    test_df.toPandas().to_csv(TEST_DF_PATH, header=True, index=False)

def create_dataset(orignal_df):
    # split text column
    final_df = orignal_df.withColumn('paragraphs', F.split(F.col('text'), "\n\n"))
    # each paragraph as row
    final_df = final_df.withColumn('rows', F.explode(F.col('paragraphs')))
    # clean
    clean_issues_udf = F.udf(preprocess, T.StringType())
    final_df = final_df.withColumn("cleaned_paragraphs", clean_issues_udf(F.col('rows')))
    # remove not required columns
    final_df = final_df.drop('text', 'paragraphs', 'rows')
    # remove empty rows and rows with only one word
    final_df = final_df.filter((F.col('cleaned_paragraphs').isNotNull()) & (F.size(F.split(F.col('cleaned_paragraphs'), "\\s+"))>1))
    # final preprocessing
    clean_paragraph_udf = F.udf(clean_para, T.StringType())
    final_df = final_df.withColumn('cleaned_paragraph', clean_paragraph_udf(F.col('cleaned_paragraphs')))
    # feature extraction
    final_df = feature_extraction(final_df, inputCol='cleaned_paragraph')
    final_df = final_df.drop('rawFeatures', 'cleaned_paragraphs')
    # save as pandas dataframe
    final_df.toPandas().to_csv(DF_PATH, header=True, index=False)

if __name__ == '__main__':
    
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    # read the parquet
    df = spark.read.parquet(PARQUET_PATH)
    df = df.filter(F.year('date')==1869).select('series', 'issue', 'date', 'text')
    # read labels as csv file
    csv_df = spark.read.csv(LABELS_CSV, header=True, inferSchema=True).select('contents', 'series')
    csv_df = labels_df(csv_df)
    # perform join
    df = df.join(csv_df, on='series', how='inner')
    dataset = create_dataset(df)
    # train-test split
    perform_split()
