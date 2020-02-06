import string
import sklearn
from nltk.classify import SklearnClassifier
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
########################################################
def preProcess(text):
    tokens=[]
    if(pd.isnull(text)==False):
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens=text.split()
    return tokens

def toVector(tokens):
    v = {}
    for t in tokens:
        try:
            v[t] +=1
        except KeyError:
            v[t] = 1
    return v
########################################################
traintext=[]
label=[]
testtext=[]

##stopword
stop_words= []
with open("stopwords-fa.txt" ,encoding = 'utf-8' ) as f:
    stop_words = f.read().splitlines()
stopwords = stop_words [1:]

train = pd.read_csv('train.csv',encoding='utf-8')
train['text']=train['comment']+" "+train['title']
for index, row in train.iterrows():
    traintext.append((toVector(preProcess(row['text'])),row['verification_status']))
    label.append(row['verification_status'])

test = pd.read_csv('test.csv',encoding='utf-8')
test['text']=test['comment']+" "+test['title']
for index,row in test.iterrows():
    testtext.append(toVector(preProcess(row['text'])))

clf=SklearnClassifier(MultinomialNB()).train(traintext)
testPredicte = clf.classify_many(testtext)
verification=pd.DataFrame({"id":test['id'],"verification_status":testPredicte ,})
verification.to_csv("ans.csv",index=False)