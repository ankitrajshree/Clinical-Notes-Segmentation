# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:02:35 2018

@author: ankit
"""

import pandas as pd
from collections import Counter

path = 'C:\\MyStuff\\ASU\\Spring_2018\\NLP\\Project\\Clinical_NLP\\Dataset\\CleanedHeaderText\\'

df1_11 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_1_11.csv')
df12_22 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et12-22.csv')
df23_27 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_batch23To27.csv')
df28 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_28.csv')
df29 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_29.csv')
df30 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_30.csv')
df31 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_31.csv')
df32 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_32.csv')
df33 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et33.csv')
df34 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_batch34.csv')
df35 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_batch35.csv')
df37 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_batch37.csv')
df38_39 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_batch38_39.csv')
df40 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_batch40.csv')
df41 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_batch41.csv')
dfi2b2 = pd.read_csv(path+'cleanedHeaderText_withoutEn2Et_i2b2.csv')

frames = [df1_11,df12_22,df23_27,df28,df29,df30,df31,df32,df33,df34,df35,df37,df38_39,df40,df41,dfi2b2]

comDf = pd.concat(frames)
pd.DataFrame.to_csv(comDf,path+"allClinicalNotesCleanedText.csv")


seq=comDf['Order of Section Header Appearence']

cnt=Counter(str(e) for e in seq)
df = pd.DataFrame.from_dict(cnt, orient='index').reset_index()
perSeqCount=df
#pd.DataFrame.to_csv(required_df,path+"cleanedHeaderText_withoutEn2Et.csv")
pd.DataFrame.to_csv(df,path+"allClinicalNotesSeqCount_withoutEn2Et.csv")
