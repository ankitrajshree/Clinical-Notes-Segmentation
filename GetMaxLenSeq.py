# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:08:43 2018

@author: ankit
"""
import pandas as pd
from collections import Counter
import ast

basePath = 'C:\\MyStuff\\ASU\\Spring_2018\\NLP\\Project\\Clinical_NLP\\Dataset\\'
data = pd.read_csv(basePath+'Train_Test_SectionHeader_text.csv')

print(data.head(1))

cnt=Counter(str(e) for e in data['Order of Section Header Appearence'].tolist())
df = pd.DataFrame.from_dict(cnt, orient='index').reset_index()
pd.DataFrame.to_csv(df,basePath+"allClinicalNotesCleanedText_SeqCnt.csv")

def getSeqLen(seqList):
    seqListofList = ast.literal_eval(seqList) 
    return len(seqListofList)

df['Seqlen'] = df['index'].apply(getSeqLen)

dfSeqLen30 = df[df['Seqlen'] > 30]

print(dfSeqLen30[0].sum())
