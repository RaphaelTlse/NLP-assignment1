import sys
sys.path.append("..")
from common.download_utils import download_week1_resources

download_week1_resources()

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from ast import literal_eval
import pandas as pd
import numpy as np

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data

train = read_data('data/train.tsv')
validation = read_data('data/validation.tsv')
test = pd.read_csv('data/test.tsv', sep='\t')
train.head()
X_train, y_train = train['title'].values, train['tags'].values
X_val, y_val = validation['title'].values, validation['tags'].values
X_test = test['title'].values

import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):

    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, " ", text)
    text = re.sub(BAD_SYMBOLS_RE, "", text)

    querywords = text.split()

    resultwords  = [word for word in querywords if word.lower() not in STOPWORDS]
    text = ' '.join(resultwords)

    return text

def test_text_prepare():
    examples = ["SQL Server - any equivalent of Excel's CHOOSE function?",
                "How to free c++ memory vector<int> * arr?"]
    answers = ["sql server equivalent excels choose function", 
               "free c++ memory vectorint arr"]
    for ex, ans in zip(examples, answers):
        if text_prepare(ex) != ans:
            return "Wrong answer for the case: '%s'" % ex
    return 'Basic tests are passed.'

print(test_text_prepare())

prepared_questions = []
for line in open('data/text_prepare_tests.tsv', encoding='utf-8'):
    line = text_prepare(line.strip())
    prepared_questions.append(line)
text_prepare_results = '\n'.join(prepared_questions)

#grader.submit_tag('TextPrepare', text_prepare_results)

#X_train = [text_prepare(x) for x in X_train]
X_val = [text_prepare(x) for x in X_val]
X_test = [text_prepare(x) for x in X_test]

#X_train[:3]

#y_train = [text_prepare(y) for y in y_train]
#y_val = [text_prepare(y) for y in y_val]

#y_train[:3]

tags_counts = {}
words_counts = {}

#for tag in y_train:
#    if tag not in tags_counts:
#        tags_counts[tag] = 0 
#    tags_counts[tag] += 1

for word in X_train:
    if word not in words_counts:
        words_counts[word] = 0 
    words_counts[word] += 1

print (y_train)
print(X_train)    
#print(tags_counts)
#print(words_counts)
