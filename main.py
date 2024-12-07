import numpy as np
import re

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer
from pyspark.ml.linalg import Vectors, VectorUDT
from sentence_transformers import SentenceTransformer

from analysis import analyze_predictions
from consts import (
    NLTK_WORDS, 
    CUSTOM_WORDS,
    PARQUET_PATH,
    LABELS_CSV,
    SPARK_APP,
    SENTENCE_TRANSFORMER,
    MODEL_PATH
)

STOPWORDS = list(set(NLTK_WORDS+CUSTOM_WORDS))
LR = LogisticRegression(featuresCol = 'features', labelCol = 'issue_label', family = 'multinomial')
EVALUATOR = MulticlassClassificationEvaluator(labelCol='issue_label', predictionCol='prediction')
LOADED_LR = LogisticRegressionModel.load(MODEL_PATH)
MODEL = SentenceTransformer(SENTENCE_TRANSFORMER, trust_remote_code=True)

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
        word = word.strip()
        if word:
            sent += word + ", "
    return sent

def embed_paragraphs(paragraph):
    paragraph = [word for word in paragraph.split()]
    return MODEL.encode(paragraph).tolist()

def agg_embeddings(features):
    return np.array(features).max(axis=0).tolist()

def train_lr_model(train_df, test_df):
    # train the model
    model = LR.fit(train_df)
    # save model
    model.write().overwrite().save(MODEL_PATH)
    # make prediction
    return model.transform(test_df)

def clean_baseline(content):
    content = ' '.join(content)
    # remove non-alphabetical characters except commas and spaces
    content = re.sub(r'[^a-zA-Z,\s]', '', content)
    # remove extra spaces around commas
    content = re.sub(r'\s*,\s*', ',', content)
    # remove extra spaces between words and convert to lower case
    content = re.sub(r'\s+', ' ', content)
    filtered_words = [word for word in content.split(',') if word not in STOPWORDS]
    # remove empty string from list
    filtered_words[:] = [word for word in filtered_words if word]
    content = ' '.join(word.lower() for word in filtered_words)
    # remove words upto 3 characters
    content = re.sub(r'\b\w{1,3}\b', '', content)
    return content.strip()

def clean_improved(content):
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

def feature_extraction_idf(df, inputCol): 
    tokenize_data =  RegexTokenizer(inputCol=inputCol, outputCol="words", minTokenLength=1, pattern="\\W").transform(df)
    hash_data = HashingTF(inputCol="words", outputCol="rawFeatures").transform(tokenize_data)
    idfModel = IDF(inputCol="rawFeatures", outputCol="features").fit(hash_data)
    return idfModel.transform(hash_data)

def feature_extraction_paragraphs(df, fetaureCol):
    # generate embeddigs
    embed_para_udf = F.udf(embed_paragraphs, T.ArrayType(T.FloatType()))
    df = df.withColumn('rawFeatures', embed_para_udf(F.col(fetaureCol)))
    # aggregate features
    aggregate_embeddings_udf = F.udf(agg_embeddings, T.ArrayType(T.FloatType()))
    df = df.withColumn('aggFeatures', aggregate_embeddings_udf(F.col('rawFeatures')))
    # convert to VectorUDT
    to_vector_udf = F.udf(lambda x: Vectors.dense(x), VectorUDT())
    df = df.withColumn('features', to_vector_udf(F.col('aggFeatures')))
    # drop aggreagte features
    df = df.drop('aggFeatures', 'rawFeatures')
    return df

def perform_split(to_split_df):
    series_df = to_split_df.select('series').distinct()
    series_train, series_test = series_df.randomSplit([0.8, 0.2], seed=0)
    train_df, test_df = to_split_df.join(series_train, on='series', how='inner'), to_split_df.join(series_test, on='series', how='inner')
    # train on only democrat and republican
    train_df = train_df.filter(F.col('issue_label').isin(0, 1))
    # test_df = test_df.filter(F.col('issue_label').isin(0, 1))
    # train_df.toPandas().to_csv('train.csv', header=True, index=False)
    # test_df.toPandas().to_csv('test.csv', header=True, index=False)
    return train_df, test_df

def evaluate_model(predictions):
    accuracy = EVALUATOR.evaluate(predictions, {EVALUATOR.metricName: "accuracy"})
    f1 = EVALUATOR.evaluate(predictions, {EVALUATOR.metricName: "f1"})
    print(f"accuracy: {accuracy}\nf1 score: {f1}")

def create_baseline_data(orignal_df, featureCols):
    # clean
    clean_issues_udf = F.udf(clean_baseline, T.StringType())
    baseline_df = orignal_df.withColumn(featureCols, clean_issues_udf(F.col('text')))
    # feature extraction
    baseline_df = feature_extraction_idf(baseline_df, inputCol=featureCols, delimiter=None)
    # drop not required columns
    baseline_df = baseline_df.drop('rawFeatures', 'text')
    return baseline_df

def create_final_data(orignal_df):
    # text colun to string
    text_to_string_udf = F.udf(list_to_string, T.StringType())
    final_df = orignal_df.withColumn('text_as_string', text_to_string_udf(F.col('text')))
    # split text column
    final_df = final_df.withColumn('paragraphs', F.split(F.col('text_as_string'), "\n\n"))
    # each paragraph as row
    final_df = final_df.withColumn('rows', F.explode(F.col('paragraphs')))
    # clean
    clean_issues_udf = F.udf(clean_improved, T.StringType())
    final_df = final_df.withColumn("cleaned_paragraphs", clean_issues_udf(F.col('rows')))
    # remove not required columns
    final_df = final_df.drop('text_as_string', 'paragraphs', 'rows')
    # remove empty rows and rows with only one word
    final_df = final_df.filter((F.col('cleaned_paragraphs').isNotNull()) & (F.size(F.split(F.col('cleaned_paragraphs'), "\\s+"))>1))
    # final preprocessing
    clean_paragraph_udf = F.udf(clean_para, T.StringType())
    final_df = final_df.withColumn('cleaned_paragraph', clean_paragraph_udf(F.col('cleaned_paragraphs')))
    # feature extraction
    # final_df = feature_extraction_paragraphs(df=final_df, inputCol=featureCols)
    final_df = feature_extraction_idf(final_df, inputCol='cleaned_paragraph')
    final_df = final_df.drop('rawFeatures', 'cleaned_paragraphs')
    return final_df

def baseline(df):
    # baseline model data
    baseline_df = create_baseline_data(df, featureCols="cleaned_content")
    # split into train and test data
    train, test = perform_split(baseline_df)
    # train lr model
    preds = train_lr_model(train, test)
    # model evaluation
    evaluate_model(predictions=preds)

def final(df):
    # final data for analysis
    final_df = create_final_data(df)
    # split into train and test data
    train, test = perform_split(final_df)
    # train lr model
    # train = spark.read.csv('train.csv', header=True, inferSchema=True)
    # test = spark.read.csv('test.csv', header=True, inferSchema=True)
    preds = train_lr_model(train, test)
    # model evaluation
    evaluate_model(predictions=preds)
    # make predictions
    predictions = LOADED_LR.transform(final_df)
    # group the predictions by issue and remove unnecessary columns
    predictions = predictions.drop('rawPrediction')
    analyze_predictions(predictions=predictions)


if __name__ == '__main__':
    
    spark = SparkSession.builder.appName(SPARK_APP).getOrCreate()
    # read the parquet
    df = spark.read.parquet(PARQUET_PATH)
    df = df.filter(F.year('date')==1869).groupBy('series', 'issue').agg(F.collect_list('text').alias('text'))
    # read labels csv file
    csv_df = spark.read.csv(LABELS_CSV, header=True, inferSchema=True).select('contents', 'series')
    csv_df = labels_df(csv_df)
    # perform join
    df = df.join(csv_df, on='series', how='inner')

    baseline(df)
    final(df)
