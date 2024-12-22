import re

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer

from consts import (
    NLTK_WORDS, 
    CUSTOM_WORDS,
    PARQUET_PATH,
    LABELS_CSV,
    SPARK_APP,
    TRAIN_DF_PATH,
    TEST_DF_PATH,
    DF_PATH,
    MODEL_PATH
)

STOPWORDS = list(set(NLTK_WORDS+CUSTOM_WORDS))
LR = LogisticRegression(featuresCol = 'features', labelCol = 'issue_label', family = 'multinomial')
EVALUATOR = MulticlassClassificationEvaluator(labelCol='issue_label', predictionCol='prediction')

def labels_df(csv_df):
    # drop na from series
    csv_df = csv_df.dropna(subset='series')
    # assign labels
    csv_df = csv_df.withColumn('label', F.when(F.array_contains(F.split(F.col('contents'), '; '), 'democratic'), 0.0).when(F.array_contains(F.split(F.col('contents'), '; '), 'republican'), 1.0).when(F.array_contains(F.split(F.col('contents'), '; '), 'independent'), 2.0).otherwise(3.0))
    csv_df = csv_df.select('series', F.col('contents').alias('metadata'), F.col('label').alias('issue_label'))
    # remove xref from series
    csv_df = csv_df.filter("series <> 'xref' AND series <> '(no report.)'")
    return csv_df

def list_to_string(ls):
    return ''.join(ls)

def clean_para(paragraph):
    sent = ''
    for word in paragraph.split(','):
        if word.strip():
            sent += word.strip() + ", "
    return sent

def feature_extraction(df, inputCol): 
    tokenize_data =  RegexTokenizer(inputCol=inputCol, outputCol="words", minTokenLength=1, pattern="\\W").transform(df)
    hash_data = HashingTF(inputCol="words", outputCol="rawFeatures").transform(tokenize_data)
    idfModel = IDF(inputCol="rawFeatures", outputCol="features").fit(hash_data)
    return idfModel.transform(hash_data)

def preprocess(content):
    if content.strip():
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
    
    return content

def train_classifier():
    # load train and test datasets
    train = spark.read.csv(TRAIN_DF_PATH, header=True, inferSchema=True)
    test = spark.read.csv(TEST_DF_PATH, header=True, inferSchema=True)
    # train Logistic Regression classifier
    model = LR.fit(train)
    # save model
    model.write().overwrite().save(MODEL_PATH)
    # make prediction
    predictions = model.transform(test)
    # evaluate performance
    accuracy, f1 = EVALUATOR.evaluate(predictions, {EVALUATOR.metricName: "accuracy"}), EVALUATOR.evaluate(predictions, {EVALUATOR.metricName: "f1"})
    print(f"accuracy: {accuracy}\nf1 score: {f1}")

def perform_split():
    to_split_df = spark.read.csv(DF_PATH, header=True, inferSchema=True)
    series_df = to_split_df.select('series').distinct()
    series_train, series_test = series_df.randomSplit([0.8, 0.2], seed=0)
    train_df, test_df = to_split_df.join(series_train, on='series', how='inner'), to_split_df.join(series_test, on='series', how='inner')
    # train on only democrat and republican
    train_df = train_df.filter(F.col('issue_label').isin(0, 1))
    # save as pandaas dataframe
    train_df.toPandas().to_csv(TRAIN_DF_PATH, header=True, index=False)
    test_df.toPandas().to_csv(TEST_DF_PATH, header=True, index=False)

def create_dataset(orignal_df):
    # text colun to string
    text_to_string_udf = F.udf(list_to_string, T.StringType())
    final_df = orignal_df.withColumn('text_as_string', text_to_string_udf(F.col('text')))
    # split text column
    final_df = final_df.withColumn('paragraphs', F.split(F.col('text_as_string'), "\n\n"))
    # each paragraph as row
    final_df = final_df.withColumn('rows', F.explode(F.col('paragraphs')))
    # clean
    clean_issues_udf = F.udf(preprocess, T.StringType())
    final_df = final_df.withColumn("cleaned_paragraphs", clean_issues_udf(F.col('rows')))
    # remove not required columns
    final_df = final_df.drop('text_as_string', 'paragraphs', 'rows')
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
    df = df.filter(F.year('date')==1869).groupBy('series', 'issue').agg(F.collect_list('text').alias('text'))
    # read labels as csv file
    csv_df = spark.read.csv(LABELS_CSV, header=True, inferSchema=True).select('contents', 'series')
    csv_df = labels_df(csv_df)
    # perform join
    df = df.join(csv_df, on='series', how='inner')
    dataset = create_dataset(df)
    # train-test split
    perform_split()
    # TODO: uncomment if you require training the model
    # train_classifier()
