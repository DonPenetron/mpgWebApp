from django.shortcuts import render
from django.http import HttpResponse
from django.templatetags.static import static

from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, LogisticRegression

import gensim
from gensim.models import FastText
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import pickle
import os
import spacy

# Create your views here.
def index(request):
    context = {'a': 'Hello World'}
    return render(request, 'index.html', context)
    #return HttpResponse({'a':1})
    
def predict(request):
    
    reg_path = os.path.join(os.path.dirname(__file__), 'reg_model.pkl')
    with open(reg_path,'rb') as f:
        reg =  pickle.load(f)

    clf_path = os.path.join(os.path.dirname(__file__), 'clf_model.pkl')
    with open(clf_path,'rb') as f:
        clf =  pickle.load(f)

    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler_model.pkl')    
    with open(scaler_path,'rb') as f:
        scaler =  pickle.load(f)

    model_path = os.path.join(os.path.dirname(__file__), 'model_model')
    fname = get_tmpfile(model_path)
    model = FastText.load(fname)
    
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    not_stop_words = set(['not'])
    stop_words -= not_stop_words
    
    if request.method == 'POST':
        text = str(request.POST.get('review'))
    
    tokenized_text = nlp(text)
    cleaned_tokenized_text = [token.lemma_ for token in tokenized_text if token.lemma_ not in stop_words]
    vectorized_text = model.infer_vector(cleaned_tokenized_text)
    
    vectorized_text = vectorized_text.reshape(1, -1)

    scaled_text = scaler.transform(vectorized_text)

    rating = reg.predict(scaled_text)[0]
    sentiment = clf.predict(scaled_text)
    
    if sentiment == 1:
        sentiment = "positive"
    elif sentiment == 0:
        sentiment = "negative"
        
    if rating < 1:
        rating = 1
    elif rating > 10:
        rating = 10
    else:
        rating = round(rating)
    
    prediction = {'rating': rating,
               'sentiment': sentiment}
    return render(request, 'index.html', prediction)