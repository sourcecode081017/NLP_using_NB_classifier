#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 22:52:44 2018

@author: anirudh
"""
import time               
start_time = time.time()  
import os,os.path
import re
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import words
from os import listdir
import numpy as np

y_true = [0]*500 + [1]*500 +  [2]*500 + [3]*500 + [4]*500 + [5]*500 + [6]*500 + [7]*500 + [8]*500 + [9]*500 + [10]*500 + [11]*500 + [12]*500 + [13]*500 + [14]*500 + [15]*500 + [16]*500 + [17]*500 + [18]*500 + [19]*497
words = (set(words.words())) #Set of words in English Language
stop_words = (set(stopwords.words('english'))) #Set of stop words
#Count of Vocabulary in the training set
VOCABULARY = 105657
proby = []

#Calculation of Prior probability
path = '20_newsgroups/train_data/'
categoryList = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]
numdocbycategory = []
for i in range(0,len(categoryList)):
    DIR = '20_newsgroups/train_data/'+categoryList[i]
    numdocbycategory.insert(i,len(os.listdir(DIR)))
    proby.insert(i,numdocbycategory[i]/10000)
    
#Preprocess test files
def testPreprocessor(FILE):
    str_topic = str(FILE.read())
    cleaned_str=re.sub('[^a-z\s]+',' ',str_topic,flags=re.IGNORECASE)
    cleaned_str=re.sub(' +',' ',cleaned_str)
    cleaned_str=cleaned_str.lower()
    cleaned_str_arr = cleaned_str.split(' ')
    cleaned_str_arr_filtered = []
    for x in cleaned_str_arr:
        if x in words:
            cleaned_str_arr_filtered.append(x)
    cleaned_str_arr_filtered_final = []
    for y in cleaned_str_arr_filtered:
        if y not in stop_words:
            
            cleaned_str_arr_filtered_final.append(y)
    cleaned_str_arr_filtered_final = list(set(cleaned_str_arr_filtered_final))
    return cleaned_str_arr_filtered_final
#Preprocess train files
def trainPreprocessor(FILE):
    str_bytes = str(FILE.read())
    cleaned_str=re.sub('[^a-z\s]+',' ',str_bytes,flags=re.IGNORECASE)
    cleaned_str=re.sub(' +',' ',cleaned_str)
    cleaned_str_n = cleaned_str.replace(' n',' ')
    cleaned_str_n=re.sub(' +',' ',cleaned_str_n)
    cleaned_str_n=cleaned_str_n.lower()
    cleaned_str_arr = cleaned_str_n.split(' ')
    cleaned_str_arr_filtered = []
    for x in cleaned_str_arr:
        if x in words:
            cleaned_str_arr_filtered.append(x)
    cleaned_str_arr_filtered_final = []
    for y in cleaned_str_arr_filtered:
        if y not in stop_words:
            
            cleaned_str_arr_filtered_final.append(y)
    key_value = Counter(cleaned_str_arr_filtered_final)
    k,v = key_value.keys(),key_value.values()
    train_dict = dict(zip(k,v))
    return  train_dict
      
        

    
#Get files names of all test documents and store it in corresponding lists    
atheism = sorted(listdir('20_newsgroups/test_data/alt.atheism'))
graphics = sorted(listdir('20_newsgroups/test_data/comp.graphics'))
mswindows = sorted(listdir('20_newsgroups/test_data/comp.os.ms-windows.misc'))
ibm =  sorted(listdir('20_newsgroups/test_data/comp.sys.ibm.pc.hardware'))
machardware = sorted(listdir('20_newsgroups/test_data/comp.sys.mac.hardware'))
windowsx = sorted(listdir('20_newsgroups/test_data/comp.windows.x'))
forsale = sorted(listdir('20_newsgroups/test_data/misc.forsale'))
autos = sorted(listdir('20_newsgroups/test_data/rec.autos'))
motorcycles = sorted(listdir('20_newsgroups/test_data/rec.motorcycles'))
baseball = sorted(listdir('20_newsgroups/test_data/rec.sport.baseball'))
hockey = sorted(listdir('20_newsgroups/test_data/rec.sport.hockey'))
crypt = sorted(listdir('20_newsgroups/test_data/sci.crypt'))
electronics = sorted(listdir('20_newsgroups/test_data/sci.electronics'))
med = sorted(listdir('20_newsgroups/test_data/sci.med'))
space = sorted(listdir('20_newsgroups/test_data/sci.space'))
guns = sorted(listdir('20_newsgroups/test_data/talk.politics.guns'))
mideast = sorted(listdir('20_newsgroups/test_data/talk.politics.mideast'))
religion = sorted(listdir('20_newsgroups/test_data/talk.religion.misc'))
politics = sorted(listdir('20_newsgroups/test_data/talk.politics.misc'))
christian = sorted(listdir('20_newsgroups/test_data/soc.religion.christian'))
test_files_list = atheism + graphics + mswindows + ibm + machardware + windowsx + forsale + autos + motorcycles + baseball + hockey + crypt + electronics + med + space + guns + mideast + religion + politics + christian 

testX = []
#Get files from each category of y_true
for i in range(0,len(test_files_list)):
    filename = ''
    filename = test_files_list[i]
    if(i>=0 and i<500):
        filepath = "20_newsgroups/test_data/alt.atheism/"+filename
    elif(i>=500 and i<1000):
        filepath = "20_newsgroups/test_data/comp.graphics/"+filename
    elif(i>=1000 and i<1500):
        filepath = "20_newsgroups/test_data/comp.os.ms-windows.misc/"+filename
    elif(i>=1500 and i<2000):
        filepath = "20_newsgroups/test_data/comp.sys.ibm.pc.hardware/"+filename
    elif(i>=2000 and i<2500):
        filepath = "20_newsgroups/test_data/comp.sys.mac.hardware/"+filename
    elif(i>=2500 and i<3000):
        filepath = "20_newsgroups/test_data/comp.windows.x/"+filename
    elif(i>=3000 and i<3500):
        filepath = "20_newsgroups/test_data/misc.forsale/"+filename
    elif(i>=3500 and i<4000):
        filepath = "20_newsgroups/test_data/rec.autos/"+filename
    elif(i>=4000 and i<4500):
        filepath = "20_newsgroups/test_data/rec.motorcycles/"+filename
    elif(i>=4500 and i<5000):
        filepath = "20_newsgroups/test_data/rec.sport.baseball/"+filename
    elif(i>=5000 and i<5500):
        filepath = "20_newsgroups/test_data/rec.sport.hockey/"+filename
    elif(i>=5500 and i<6000):
        filepath = "20_newsgroups/test_data/sci.crypt/"+filename
    elif(i>=6000 and i<6500):
        filepath = "20_newsgroups/test_data/sci.electronics/"+filename
    elif(i>=6500 and i<7000):
        filepath = "20_newsgroups/test_data/sci.med/"+filename
    elif(i>=7000 and i<7500):
        filepath = "20_newsgroups/test_data/sci.space/"+filename
    elif(i>=7500 and i<8000):
        filepath = "20_newsgroups/test_data/talk.politics.guns/"+filename
    elif(i>=8000 and i<8500):
        filepath = "20_newsgroups/test_data/talk.politics.mideast/"+filename
    elif(i>=8500 and i<9000):
        filepath = "20_newsgroups/test_data/talk.religion.misc/"+filename
    elif(i>=9000 and i<9500):
        filepath = "20_newsgroups/test_data/talk.politics.misc/"+filename
    else:
        filepath = "20_newsgroups/test_data/soc.religion.christian/"+filename
    FILE = open(filepath,"rb")
    testX.append(sorted(testPreprocessor(FILE)))

#Get list of train words from train mega document
train_word_list = []
MF1 = open("20_newsgroups/traindatam/alt.atheism/megaatheism","rb")
train_word_list.append(trainPreprocessor(MF1))
MF2 = open("20_newsgroups/traindatam/comp.graphics/megacomgraphics","rb")
train_word_list.append(trainPreprocessor(MF2))
MF3 = open("20_newsgroups/traindatam/comp.os.ms-windows.misc/megaoswindows","rb")
train_word_list.append(trainPreprocessor(MF3))
MF4 = open("20_newsgroups/traindatam/comp.sys.ibm.pc.hardware/megaibm","rb")
train_word_list.append(trainPreprocessor(MF4))
MF5 = open("20_newsgroups/traindatam/comp.sys.mac.hardware/megamachardware","rb")
train_word_list.append(trainPreprocessor(MF5))
MF6 = open("20_newsgroups/traindatam/comp.windows.x/megawindowsx","rb")
train_word_list.append(trainPreprocessor(MF6))
MF7 = open("20_newsgroups/traindatam/misc.forsale/megamiscforsale","rb")
train_word_list.append(trainPreprocessor(MF7))
MF8 = open("20_newsgroups/traindatam/rec.autos/megaautos","rb")
train_word_list.append(trainPreprocessor(MF8))
MF9 = open("20_newsgroups/traindatam/rec.motorcycles/megamotorcycle","rb")
train_word_list.append(trainPreprocessor(MF9))
MF10 = open("20_newsgroups/traindatam/rec.sport.baseball/megabaseball","rb")
train_word_list.append(trainPreprocessor(MF10))
MF11 = open("20_newsgroups/traindatam/rec.sport.hockey/megahockey","rb")
train_word_list.append(trainPreprocessor(MF11))
MF12 = open("20_newsgroups/traindatam/sci.crypt/megacrypt","rb")
train_word_list.append(trainPreprocessor(MF12))
MF13 = open("20_newsgroups/traindatam/sci.electronics/megaelectronics","rb")
train_word_list.append(trainPreprocessor(MF13))
MF14 = open("20_newsgroups/traindatam/sci.med/megamed","rb")
train_word_list.append(trainPreprocessor(MF14))
MF15 = open("20_newsgroups/traindatam/sci.space/megaspace","rb")
train_word_list.append(trainPreprocessor(MF15))
MF16 = open("20_newsgroups/traindatam/talk.politics.guns/megaguns","rb")
train_word_list.append(trainPreprocessor(MF16))
MF17 = open("20_newsgroups/traindatam/talk.politics.mideast/megamideast","rb")
train_word_list.append(trainPreprocessor(MF17))
MF18 = open("20_newsgroups/traindatam/talk.religion.misc/megareligionmisc","rb")
train_word_list.append(trainPreprocessor(MF18))
MF19 = open("20_newsgroups/traindatam/talk.politics.misc/megapoliticsmisc","rb")
train_word_list.append(trainPreprocessor(MF19))
MF20 = open("20_newsgroups/traindatam/soc.religion.christian/megachristian","rb")
train_word_list.append(trainPreprocessor(MF20))

#The Naive bayes classifier algorithm
def nbClassifier(test_word_list,train_word_dict,j):
    probij = []
    prior = 0
    for x in range(0,len(test_word_list)):
        cnt = train_word_dict.get(test_word_list[x],0)
        probij.append(np.log(((cnt + 1)/((sum(train_word_dict.values())) + VOCABULARY))))
    prior = np.sum(probij) + np.log(proby[j])
    return prior
     
        
#Predict category for each document in test and print it       
predict = []       
for i in range(0,9997):#For every document in test
    probabilities = [] # Initialize prob list for every document
    for j in range(0,len(train_word_list)):#for every word list in train
        probabilities.append(nbClassifier(testX[i],train_word_list[j],j))
    predict.append(np.argmax(probabilities)) #argmax of probabilities list is the predicted value
    print(predict[i] , y_true[i])
print('The prediction vector is', predict)

#Calculate and print accuracy
res = [ai - bi for ai,bi in zip(y_true,predict)]
err = np.count_nonzero(res)/9997
acc = 1-err
print('accuracy of this NB Model = ',acc)
print('Execution time of program in mins = ',(time.time()-start_time)/60) 




