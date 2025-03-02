SPARK_APP = "PartisianClassification"
# TODO: REPLACE WITH PATH OF YOUR PARQUET FILE
PARQUET_PATH = ""
# TODO: REPLACE WITH PATH OF YOUR METADATA FILE
LABELS_CSV = ""
# TODO: REPLACE WITH YOUR MODEL DIRECTORY
MODEL_PATH = ""
# TODO: REPLACE WITH PATH OF MASTER DATAFRAME
DF_PATH = ""
# TODO: REPLACE WITH PATH OF TRAIN AND TEST DATAFRAMES
TRAIN_DF_PATH = ""
TEST_DF_PATH = ""
# TODO: REPLACE WITH PATH OF PREDICTIONS DATAFRAME
PREDICTIONS_DF_PATH = ""
# TODO: REPLACE WITH PATH OF DEMOCRAT AND REPUBLICAN DATAFRAMES
DEM_PATH = ""
REP_PATH = ""
# TODO: REPLACE WITH PATH OF HISTOGRAMS
HIST_FOLDER = ""

NLTK_WORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", 
    "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", 
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", 
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", 
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", 
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", 
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", 
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
]

CUSTOM_WORDS = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against", "al", "all", "almost", "alone", 
    "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "analyze", "and", "another", 
    "any", "anyhow", "anyone", "anything", "anywhere", "applicable", "apply", "are", "around", "as", "assume", "at", 
    "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "being", "below", 
    "beside", "besides", "between", "beyond", "both", "but", "by", "came", "cannot", "cc", "cm", "come", "compare", 
    "could", "de", "dealing", "department", "depend", "did", "discover", "dl", "do", "does", "during", "each", "ec", 
    "ed", "effected", "eg", "either", "else", "elsewhere", "enough", "et", "etc", "ever", "every", "everyone", 
    "everything", "everywhere", "except", "find", "for", "found", "from", "further", "get", "give", "go", "gov", 
    "had", "has", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", 
    "him", "himself", "his", "how", "however", "hr", "ie", "if", "ii", "iii", "in", "inc", "incl", "indeed", "into", 
    "investigate", "is", "it", "its", "itself", "j", "jour", "journal", "just", "kg", "last", "latter", "latterly", 
    "lb", "ld", "letter", "like", "ltd", "made", "make", "many", "may", "me", "meanwhile", "mg", "might", "ml", "mm", 
    "mo", "more", "moreover", "most", "mostly", "mr", "much", "must", "my", "myself", "namely", "neither", "never", 
    "nevertheless", "next", "no", "nobody", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", 
    "on", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", 
    "oz", "per", "perhaps", "pm", "precede", "presently", "previously", "pt", "rather", "regarding", "relate", "said", 
    "same", "seem", "seemed", "seeming", "seems", "seriously", "several", "she", "should", "show", "showed", "shown", 
    "since", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "studied", 
    "sub", "such", "take", "tell", "th", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", 
    "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "thorough", "those", "though", 
    "through", "throughout", "thru", "thus", "to", "together", "too", "toward", "towards", "try", "type", "ug", "under", 
    "unless", "until", "up", "upon", "us", "used", "using", "various", "very", "via", "was", "we", "were", "what", 
    "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", 
    "wherever", "whether", "which", "while", "whither", "who", "whoever", "whom", "whose", "why", "will", "with", 
    "within", "without", "wk", "would", "wt", "yet", "you", "your", "yours", "yourself", "yourselves", "yr"
]
