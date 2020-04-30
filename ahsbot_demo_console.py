# |-----------------------------------------------------------------------------
# |            This source code is provided under the Apache 2.0 license      --
# |  and is provided AS IS with no warranty or guarantee of fit for purpose.  --
# |                See the project's LICENSE.md for details.                  --
# |           Copyright Refinitiv 2020. All rights reserved.                  --
# |-----------------------------------------------------------------------------

# |-----------------------------------------------------------------------------
# |          Refinitiv Messenger BOT API via HTTP REST and WebSocket          --
# |-----------------------------------------------------------------------------

from nltk.stem.lancaster import LancasterStemmer
import nltk
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import warnings
import sys
import time
import getopt
import requests
import socket
import json
import websocket
import threading
import random
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Set user name
userId="moragodkrit"

# Set intents file with the file to save data and model
intents_filename = 'ahsbot_intents.json'
pickle_model_filename = 'ahsbot-model.pkl'
pickle_data_filename = 'ahsbot-data.pkl'

# Setting Log level the supported value is 'logging.WARN' and 'logging.DEBUG'
log_level = logging.WARN


# Chatroom objects
chatroom_id = None

# Control app flow set it to false to stop
is_running = True

# Input your Eikon Data API AppKey
eikon_data_api_app_key = "1f67bc1210a2464ca06128c749258abf68ea6eee"

use_data_from_eikon =True
if use_data_from_eikon:
    #For Eikon Data API
    import eikon as ek

# ========== For Machine Learning ====================
warnings.filterwarnings("ignore")


# things we need for NLP
stemmer = LancasterStemmer()

# import our chat-bot intents file
intents = None
model = None
words = None
classes = None
data = None
# Default Confidence Level to filter result
confidence_level = 0.85


def loadmodel():
    global intents
    global intents_filename
    global pickle_data_filename
    global pickle_model_filename
    global data
    global words
    global classes
    with open(intents_filename) as json_data:
        intents = json.load(json_data)

    data = pickle.load(open(pickle_data_filename, "rb" ) )
    words = data['words']
    classes = data['classes']
    # Use pickle to load in the pre-trained model

    with open(pickle_model_filename, 'rb') as f:
        global model
        model = pickle.load(f)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return(np.array(bag))


def classify(sentence):
    # Add below two lines for workaround error _thread._local' object has no attribute 'value'
    import keras.backend.tensorflow_backend as tb
    tb._SYMBOLIC_SCOPE.value = True
    ################################
    global model
    global confidence_level
    ERROR_THRESHOLD = confidence_level
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)],
                              dtype=float, index=['input'])
    results = model.predict([input_data], workers=1, verbose=0)[0]

    # filter out predictions below a threshold, and provide intent index
    results = [[i, r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # If there is no match result manually return no answer or do not understand
    if not return_list:
        #print("no answer")
        return_list.append(("noanswer", "1.0"))
    # return tuple of intent and probability

    return return_list


# ========================= Managed to request data =============================

def getsnapshot(ricList):
    message = ""
    if use_data_from_eikon:
        try:
            data, err = ek.get_data([ricList], ["TR.Open","TR.PriceClose", "TR.Volume", "TR.PriceLow"])
            if err:
                message = "Sorry unable to retreive data for {} {}".format(ricList,err[0]['message'])
            else:
                message = "The lastest data for {} are as follow\n {}".format(ricList,data.to_string())
                
        except ek.EikonError as ex:
            print("Get Snapshot Error:{}".format(ex.message))
            message="Sorry, I'm unable to retreive data. {}".format(ex.message)
    else:
        message= 'Dummy price data for {} are as follows\nInstrument  Price Open  Price Close    Volume  Price Low\n'\
                 '0     {}      172.06       174.55  34305320     170.71'.format(ricList,ricList)
    return message
    pass


def gettimeseries(ricList):
    message = ""
    if use_data_from_eikon :
        from eikon.tools import get_date_from_today
        try:
            df = ek.get_timeseries([ricList], start_date= get_date_from_today(30), end_date = get_date_from_today(0), interval="daily")
            message = "Timeseries data for {} are as follows\n{}".format(ricList,df.to_string())
        except ek.EikonError as err:
            message = 'Sorry unable to retreive timeseries data {}'.format(err.message)
    else:
        message = 'Dummy Timeseries data for {} are as follows\n'\
            '{}           HIGH      LOW    OPEN   CLOSE    COUNT       VOLUME\n'\
            'Date                                                             \n'\
            '2020-03-30  117.9800  111.200  116.64  114.34  17651.0   68698753.0\n'\
            '2020-03-31  117.7915  111.640  116.00  113.00  22291.0  109949486.0\n'\
            '2020-04-01  112.3000  108.200  109.42  109.60  21952.0  105886576.0\n'\
            '2020-04-02  112.0200  108.000  110.00  110.30  19332.0  106155107.0\n'\
            '2020-04-03  111.1600  107.480  109.98  111.02  12003.0   59174173.0\n'\
            '2020-04-06  116.7200  111.260  111.40  116.56  13808.0   71029106.0\n'\
            '2020-04-07  121.0200  114.880  118.84  115.68  17768.0   77232492.0\n'\
            '2020-04-08  116.6000  109.780  115.52  110.84  21285.0  100841744.0\n'\
            '2020-04-09  114.0800  110.800  112.36  113.02  16333.0   82425809.0\n'\
            '2020-04-14  114.7400  111.480  114.24  111.98  13815.0   82331863.0\n'\
            '2020-04-15  113.3000  107.800  112.78  108.00  15805.0  105933820.0\n'\
            '2020-04-16  109.2000  106.220  108.74  107.16  17415.0   69673185.0\n'\
            '2020-04-17  110.8800  108.040  109.92  108.88  13169.0   68124771.0\n'\
            '2020-04-20  112.0000  108.773  110.38  111.00  15398.0   83415607.0\n'\
            '2020-04-21  110.1200  105.860  109.84  105.86  15231.0   69159078.0\n'\
            '2020-04-22  107.6400  105.380  106.82  107.22  10462.0   47750579.0\n'\
            '2020-04-23  109.0400  105.500  107.06  107.72  11011.0   96087921.0\n'\
            '2020-04-24  110.0400  105.520  106.68  109.32  12717.0   59498404.0\n'\
            '2020-04-27  111.6600  110.040  110.88  110.20   2991.0    8430558.0\n'.format(ricList,ricList)

    return message
    pass

# ========================================================================

# Datastructure to keep the context and contextAction
context = {}
contextAction = {}


def botResponse(sentence, userID):
    import random
    currentContext = ''
    useDialog = False
    if (userID in context) and (context[userID] != ''):
        currentContext = context[userID]
        useDialog = True
    tagResult = classify(sentence)
    # Find JSON intent
    responseList = list(filter(lambda x: x['tag'] == tagResult[0][0], intents['intents']))
    if tagResult[0][0] =="goodbye":
        global is_running
        is_running=False
        return random.choice(responseList[0]['responses'])

    if tagResult[0][0] == 'clear_context':
        if userID in context:
            context.pop(userID)
        return random.choice(responseList[0]['responses'])

    if not useDialog and responseList:
        if responseList[0]['context']:
            # context should contains only 1 member
            context[userID] = responseList[0]['context'][0]
        else:
            context.pop(userID)

        return random.choice(responseList[0]['responses'])
    else:
        currentContext = list(filter(lambda x: x['tag'] == context[userID], intents['intents']))
        if ('context_set' in currentContext[0]) and (tagResult[0][0] in currentContext[0]['context_set']):
            if responseList[0]['context']:
                # context should contains only 1 member
                context[userID] = responseList[0]['context'][0]
            else:
                context.pop(userID)

            return random.choice(responseList[0]['responses'])
        else:
            if 'context_link' in currentContext[0]:
                linkContext = list(filter(lambda x: x['tag'] == currentContext[0]['context_link'][0], intents['intents']))
                if linkContext[0]['context'] and linkContext[0]['context'][0] != '':
                    # context should contains only 1 member
                    context[userID] = linkContext[0]['context'][0]
                else:
                    contextAction[userID] =\
                        {
                        "actionName": context[userID],
                        "actionValue": sentence
                    }
                    context.pop(userID)
                return random.choice(linkContext[0]['responses'])
            pass
        context.pop(userID)
        return random.choice(currentContext[0]['responses'])
    pass

def process_message(incoming_msg,sender):  # Process incoming message from a joined Chatroom
    resp_msg = '@'+sender+' '+botResponse(incoming_msg, sender)
    global confidence_level
    if sender in contextAction:
        #print(contextAction)
        actionValue = contextAction[sender]['actionValue']
        if contextAction[sender]['actionName'] == "getsnapshot":
            resp_msg = '@'+sender+' '+getsnapshot(actionValue)
        elif contextAction[sender]['actionName'] == "gettimeseries":
            resp_msg = '@'+sender+' '+gettimeseries(actionValue)
        elif contextAction[sender]['actionName'] == "changeconfidencelevel":
            confidence_level = float(actionValue)
            resp_msg = 'Confidence Level is now adjusted to '+actionValue
        contextAction.pop(sender)
    return resp_msg


# =============================== Main Process ========================================


# Running the tutorial
if __name__ == '__main__':

        if use_data_from_eikon:
            try:
                # Set Eikon AppKey to reqeuest data from Eikon Desktop via Eikon Data API
                ek.set_app_key(eikon_data_api_app_key)
            except ek.EikonError as ex:
                print("Eikon Initialize Error:{}".format(ex.message))

        print('Load data and model for AHS Chatbot...')
        loadmodel()
        # Setting Python Logging
        logging.basicConfig(
            format='%(levelname)s:%(name)s :%(message)s', level=log_level)
        print("ChatRoom is starting now. Type 'bye' to exit")
        while is_running:
            message = input("{} input:".format(userId))
            print("Bot:{}\n".format(process_message(message,userId)))
