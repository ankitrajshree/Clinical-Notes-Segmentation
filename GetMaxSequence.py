# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 15:38:58 2018

@author: ankit
"""
import pandas as pd
import ast
import jellyfish as jf
#import numpy as np

basePath = 'C:\\MyStuff\\ASU\\Spring_2018\\NLP\\Project\\Clinical_NLP\\Dataset\\'

seqList = pd.read_csv(basePath + 'allClinicalNotesSeqCount_withoutEn2Et.csv', header = None)
seqList = seqList[1:]

headerMapDict = {'admission_date':'a',
 'allergies_and_adverse_reactions':'b',
 'chief_complaint':'c',
 'history_present_illness':'d',
 'review_of_systems':'e',
 'past_medical_history':'f',
 'social_history':'g',
 'family_history':'h',
 'physical_examination':'i',
 'assessment':'j',
 'laboratory_and_radiology_data':'k',
 'diagnoses':'l',
 'findings':'m',
 'hospital_course':'n',
 'medications':'o',
 'discharge_condition':'p',
 'activity':'q',
 'discharge_instructions':'r',
 'history_source':'s',
 'follow_up':'t',
 'assessment_and_plan':'u'}

def getSeqLen(seqList):
    seqListofList = ast.literal_eval(seqList) 
    return len(seqListofList)


def getSeqList(seqList):
    seqListofList = ast.literal_eval(seqList) 
    return seqListofList

print(seqList[2].max())
seqList['Seqlen'] = seqList[1].apply(getSeqLen)
#seqStr = str(maxSeqLen)
#print(seqList.loc[seqList[1] == seqStr][2])


seqListNew = seqList[seqList['Seqlen']>1]

top100ValDf = seqListNew[seqListNew[2] >= 100]
below100ValDf = seqListNew[seqListNew[2] < 100]

top100List = top100ValDf[1].tolist()
top100List = [ast.literal_eval(x) for x in top100List]

below100List = below100ValDf[1].tolist()
below100List = [ast.literal_eval(x) for x in below100List]

def getEditDistance(inp):
    global top10MapStr
    return [ jf.damerau_levenshtein_distance(x,inp) for x in top10MapStr]

def getCharRep(inp):
    return headerMapDict[inp]

def getMinList(inp):
    return min(inp)

top100MapStr = list(map(lambda x : ''.join(list(map(getCharRep,x))),top100List))
below100MapStr = list(map(lambda x : ''.join(list(map(getCharRep,x))),below100List))

editDist = list((map(getEditDistance,below100MapStr)))


# =============================================================================
# top10ValDf = seqListNew[seqListNew[2] >= 3000]
# below10ValDf = seqListNew[seqListNew[2] < 3000]
# 
# top10List = top10ValDf[1].tolist()
# top10List = [ast.literal_eval(x) for x in top10List]
# 
# below10List = below10ValDf[1].tolist()
# below10List = [ast.literal_eval(x) for x in below10List]
# 
# top10MapStr = list(map(lambda x : ''.join(list(map(getCharRep,x))),top10List))
# below10MapStr = list(map(lambda x : ''.join(list(map(getCharRep,x))),below10List))
# 
# editDist10 = list((map(getEditDistance,top10MapStr)))
# 
# =============================================================================

editDistMin = list((map(getMinList,editDist)))

minCount = sum(x > 5 for x in editDistMin)

minCountIdx = [x > 5 for x in editDistMin]

below100ValDf['EditDistMin'] = minCountIdx

toRemoveDF = below100ValDf.loc[below100ValDf['EditDistMin'] == True]




