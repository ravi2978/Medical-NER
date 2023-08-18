# !pip install flask
# !pip install pycrf
# !pip install sklearn-crfsuite
# %pip install spacy
# !pip install streamlit

import streamlit as st
import os
import pandas as pd 
import flask
import pickle
# from flask import Flask, render_template, request         #LIBRARIES USED
import spacy
import textwrap
import warnings
import json
import sklearn_crfsuite
from tqdm import tqdm
from sklearn_crfsuite import metrics

warnings.filterwarnings("ignore")
    
model = spacy.load("en_core_web_sm")
crf = pickle.load(open('pred.pkl', 'rb'))
med_dict = pickle.load( open( "save.p", "rb" ) )

def getFeaturesForOneWord(word_details, pos):
    word_details.reset_index(drop=True, inplace=True)
    word = word_details[pos][0]
    postag = word_details[pos][1]
    
    features = [
        'bias=' + "1.0",
        'word.lower=' + word.lower(),
        'word[-3]=' + word[:-3],
        'word[-2]=' + word[:-2],
        'word.islower=%s' % word.islower(),
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag.isnounpronoun=%s' % (postag in ['NOUN','PROPN']),
    ]
    
    if (pos > 0):
        prev_word = word_details[pos-1][0]
        prev_postag = word_details[pos-1][1]
        
        features.extend([
            'prev_word.lower=' + prev_word.lower(),
            'prev_word[-3]=' + prev_word[:-3],
            'prev_word[-2]=' + prev_word[:-2],
            'prev_word.islower=%s' % prev_word.islower(),
            'prev_word.isupper=%s' % prev_word.isupper(),
            'prev_word.istitle=%s' % prev_word.istitle(),
            'prev_word.isdigit=%s' % prev_word.isdigit(),
            'prev_postag=' + prev_postag,
            'prev_postag.isnounpronoun=%s' % (prev_postag in ['NOUN','PROPN']),
        ])
    else:
        features.append('BEG')
        
    if (pos < len(word_details) - 1):
        next_word = word_details[pos+1][0]
        next_postag = word_details[pos+1][1]
        
        features.extend([
            'next_word.lower=' + next_word.lower(),
            'next_word[-3]=' + next_word[:-3],
            'next_word[-2]=' + next_word[:-2],
            'next_word.islower=%s' % next_word.islower(),
            'next_word.isupper=%s' % next_word.isupper(),
            'next_word.istitle=%s' % next_word.istitle(),
            'next_word.isdigit=%s' % next_word.isdigit(),
            'next_postag=' + next_postag,
            'next_postag.isnounpronoun=%s' % (next_postag in ['NOUN','PROPN']),
        ])
    else:
        features.append('END')
        
    return features

def get_word_details(item):
    return item["text"], item["pos"]

def disease_prediction(input):
    disease=''
    treatment=''

    input_text = []
    input_pos = []
    input_label = []

    # input_sent = "primary tumor ( li ) bronchogenic carcinoma"

#     input_sent = "The `` corrected '' cesarean rate ( maternal-fetal medicine and transported patients excluded ) was 12.4 % ( 273 of 2194 ) , and the `` corrected '' primary rate was 9.6 % ( 190 of 1975 )"

    input_sent = input
    input_disease = input_sent
    input_model = model(input_disease)

    for word in input_model:
        input_text.append(word.text)
        input_pos.append(word.pos_)
        input_label.append('D')

    details_sent = pd.DataFrame({'text':input_text, 'pos':input_pos,'label':input_label})
    words_for_features = details_sent.apply(get_word_details, axis=1)
    # print(words_for_features)

    test_sent = []

    for i in range(len(input_disease.split())):
        test_sent.append(getFeaturesForOneWord(words_for_features, i))

    # print(test_sent)
    for i,tag in enumerate(crf.predict([test_sent])[0]):

        if tag == 'D':
            tr = input_disease.split()[i]
            disease += tr + " "

            if tr in med_dict:
                treatment += med_dict.get(tr) + ", "

            if disease.strip() in med_dict:
                treatment += med_dict.get(disease.strip()) + ", "

    disease = disease.strip()

    if len(treatment) == 0:
        treatment = 'Not Available'
    else:
        treatment = treatment.rstrip(", ")

    print('Identified Disease   :', disease)
    print('Identified Treatment :', treatment)
    return "Identified disease - ",disease,'Identified treatment - ',treatment

def main():
    st.title("Disease Recognition")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Disease Recognition ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    input = st.text_input("Enter Your Query","Type Here")
    result=""
    if st.button("Predict"):
        result = disease_prediction(input)
    st.success(format(result))
    

if __name__=='__main__':
    main()