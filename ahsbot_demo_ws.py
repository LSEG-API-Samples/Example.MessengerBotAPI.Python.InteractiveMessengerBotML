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
# RDPTokenManagement from rdp_token.py
from rdp_token import RDPTokenManagement


# Input your Bot Username
bot_username = '<Bot Agent UserName: botagain.user@company.com etc>'
# Input Bot Password
bot_password = '<Bot Agent Password>'
# Input your Messenger account AppKey.
app_key = '<Bot Agent App key>'

# Input your Eikon Data API AppKey
eikon_data_api_app_key = "<Eikon Data API App Key>"
# Input your Messenger Application account email
recipient_email = ['<Recipient Email: This should be your email used to login Eikon Messenger>']

# Input BILATERAL Chat Room name
bilateral_chatroom_name = "<BILATERAL CHATROOM NAME>"

# Set intents file with the file to save data and model
intents_filename = 'ahsbot_intents.json'
pickle_model_filename = 'ahsbot-model.pkl'
pickle_data_filename = 'ahsbot-data.pkl'

# Set wether or not app will retreive data from Eikon using Eikon Data API. This options for the case that user does not have Eikon.
# App will post dummy price and send it back to use.
use_data_from_eikon =True
if use_data_from_eikon:
    #For Eikon Data API
    import eikon as ek

# Setting Log level the supported value is 'logging.WARN' and 'logging.DEBUG'
log_level = logging.WARN


# Authentication and connection objects
auth_token = None
rdp_token = None
access_token = None
expire_time = 0
logged_in = False

# Chatroom objects
chatroom_id = None

# Please verify below URL is correct via the WS lookup
ws_url = 'wss://api.collab.refinitiv.com/services/nt/api/messenger/v1/stream'
gw_url = 'https://api.refinitiv.com'
bot_api_base_path = '/messenger/beta1'


# =============================== RDP and Messenger BOT API functions ========================================

def authen_rdp(rdp_token_object):  # Call RDPTokenManagement to get authentication
    # Based on WebSocket application behavior, the Authentication will not read/write Token from rest-token.txt file
    auth_token = rdp_token_object.get_token(save_token_to_file=False)
    if auth_token:
        # return RDP access token (sts_token) , expire_in values and RDP login status
        return auth_token['access_token'], auth_token['expires_in'], True
    else:
        return None, 0, False


def inviteToChatRooms(contact_mail, room_id):

    url = '{}{}/chatrooms/{}/invite'.format(
        gw_url, bot_api_base_path, room_id)
    print(url)
    body = {
        'emails': [contact_mail]
    }

    # Print for debugging purpose
    logging.debug('Sent: %s' % (json.dumps(
        body, sort_keys=True, indent=2, separators=(',', ':'))))

    response = None
    try:
        # Send a HTTP request message with Python requests module
        response = requests.post(
            url=url, data=json.dumps(body), headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: post message to exception failure:', e)

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: post message to chatroom success')
        # Print for debugging purpose
        logging.debug('Receive: %s' % (json.dumps(
            response.json(), sort_keys=True, indent=2, separators=(',', ':'))))
    else:
        print('Messenger BOT API: post message to failure:',
              response.status_code, response.reason)
        print('Text:', response.text)
    pass


def create_bilateral_chatroom(access_token, room_name):
    url = '{}{}/chatrooms'.format(gw_url, bot_api_base_path, room_name)
    print(url)
    body = {
        'name': room_name
    }

    # Print for debugging purpose
    logging.debug('Sent: %s' % (json.dumps(
        body, sort_keys=True, indent=2, separators=(',', ':'))))

    response = None
    try:
        # Send a HTTP request message with Python requests module
        response = requests.post(
            url=url, data=json.dumps(body), headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: create bilateral chatroom exception failure:{}'.format(e))

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: create bilateral chatroom name {} success'.format(room_name))
        # Print for debugging purpose
        logging.debug('Receive: %s' % (json.dumps(
            response.json(), sort_keys=True, indent=2, separators=(',', ':'))))
        print(response)
        return response.json()['room']['chatroomId']
    else:
        print('Messenger BOT API: creaet bilateral chat room failed:',
              response.status_code, response.reason)
        print('Text:', response.text)
    pass

# Get List of Chatrooms Function via HTTP REST


def list_chatrooms(access_token, room_is_managed=False):

    if room_is_managed:
        url = '{}{}/managed_chatrooms'.format(gw_url, bot_api_base_path)
    else:
        url = '{}{}/chatrooms'.format(gw_url, bot_api_base_path)

    response = None
    try:
        # Send a HTTP request message with Python requests module
        response = requests.get(
            url, headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: List Chatroom exception failure:', e)

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: get chatroom  success')
    else:
        print('Messenger BOT API: get chatroom result failure:',
              response.status_code, response.reason)
        print('Text:', response.text)

    return response.status_code, response.json()


def join_chatroom(access_token, room_id=None, room_is_managed=False):  # Join chatroom
    if room_is_managed:
        url = '{}{}/managed_chatrooms/{}/join'.format(
            gw_url, bot_api_base_path, room_id)
    else:
        url = '{}{}/chatrooms/{}/join'.format(gw_url,
                                              bot_api_base_path, room_id)
    response = None
    try:
        # Send a HTTP request message with Python requests module
        response = requests.post(
            url, headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: join chatroom exception failure:', e)

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: join chatroom success')
        # Print for debugging purpose
        logging.debug('Receive: %s' % (json.dumps(response.json(),
                                                  sort_keys=True, indent=2, separators=(',', ':'))))
    else:
        print('Messenger BOT API: join chatroom result failure:',
              response.status_code, response.reason)
        print('Text:', response.text)



# send 1 to 1 message to recipient email directly without a Chatroom via BOT
def post_direct_message(access_token, contact_email='', text=''):
    url = '{}{}/message'.format(gw_url, bot_api_base_path)

    body = {
        'recipientEmail': contact_email,
        'message': text
    }

    # Print for debugging purpose
    logging.debug('Sent: %s' % (json.dumps(
        body, sort_keys=True, indent=2, separators=(',', ':'))))

    try:
        # Send a HTTP request message with Python requests module
        response = requests.post(
            url=url, data=json.dumps(body), headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: post a 1 to 1 message exception failure:', e)

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: post a 1 to 1 message to %s success' %
              (contact_email))
        # Print for debugging purpose
        logging.debug('Receive: %s' % (json.dumps(response.json(),
                                                  sort_keys=True, indent=2, separators=(',', ':'))))
    else:
        print('Messenger BOT API: post a 1 to 1 message failure:',
              response.status_code, response.reason)
        print('Text:', response.text)


# Posting Messages to a Chatroom via HTTP REST
def post_message_to_chatroom(access_token, room_id=None,  text='', room_is_managed=False):
    if room_is_managed:
        url = '{}{}/managed_chatrooms/{}/post'.format(gw_url, bot_api_base_path, room_id)
    else:
        url = '{}{}/chatrooms/{}/post'.format(gw_url, bot_api_base_path, room_id)

    body = {
        'message': text
    }

    # Print for debugging purpose
    logging.debug('Sent: %s' % (json.dumps(body, sort_keys=True, indent=2, separators=(',', ':'))))

    response = None
    try:
        # Send a HTTP request message with Python requests module
        response = requests.post(url=url, data=json.dumps(body), headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: post message to exception failure:', e)

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: post message to chatroom success')
        # Print for debugging purpose
        logging.debug('Receive: %s' % (json.dumps(response.json(), sort_keys=True, indent=2, separators=(',', ':'))))
    else:
        print('Messenger BOT API: post message to failure:',response.status_code, response.reason)
        print('Text:', response.text)
    pass

# Remove Participant from chatroom. Room must created by bOT


def remove_participant(access_token, room_id=None, emailList="", room_is_managed=False):

    if room_is_managed:
        url = '{}{}/managed_chatrooms/{}/participants/remove'.format(
            gw_url, bot_api_base_path, room_id)
    else:
        url = '{}{}/chatrooms/{}/participants/remove'.format(
            gw_url, bot_api_base_path, room_id)
    body = {
        'emails': [emailList]
    }
    response = None
    try:
        # Send a HTTP request message with Python requests module
        response = requests.post(url, data=json.dumps(body), headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: remove participant failure:', e)

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: remove participant success')
        # Print for debugging purpose
        logging.debug('Receive: %s' % (json.dumps(response.json(), sort_keys=True, indent=2, separators=(',', ':'))))
    else:
        print('Messenger BOT API: remove participant failure:',response.status_code, response.reason)
        print('Text:', response.text)

# Leave a joined Chatroom
def leave_chatroom(access_token, room_id=None, room_is_managed=False):
    if room_is_managed:
        url = '{}{}/managed_chatrooms/{}/leave'.format(
            gw_url, bot_api_base_path, room_id)
    else:
        url = '{}{}/chatrooms/{}/leave'.format(gw_url, bot_api_base_path, room_id)
    response = None
    try:
        # Send a HTTP request message with Python requests module
        response = requests.post(url, headers={'Authorization': 'Bearer {}'.format(access_token)})
    except requests.exceptions.RequestException as e:
        print('Messenger BOT API: leave chatroom exception failure:', e)

    if response.status_code == 200:  # HTTP Status 'OK'
        print('Messenger BOT API: leave chatroom success')
        # Print for debugging purpose
        logging.debug('Receive: %s' % (json.dumps(response.json(), sort_keys=True, indent=2, separators=(',', ':'))))
    else:
        print('Messenger BOT API: leave chatroom failure:',response.status_code, response.reason)
        print('Text:', response.text)


# =============================== WebSocket functions ========================================


def on_message(_, message):  # Called when message received, parse message into JSON for processing
    print('Received: ')
    message_json = json.loads(message)
    print(json.dumps(message_json, sort_keys=True, indent=2, separators=(',', ':')))
    process_message(message_json)


def on_error(_, error):  # Called when websocket error has occurred
    print(error)


def on_close(_):  # Called when websocket is closed
    print('WebSocket Connection Closed')
    leave_chatroom(access_token, chatroom_id)


def on_open(_):  # Called when handshake is complete and websocket is open, send login
    print('WebSocket Connection is established')
    # Send "connect command to the WebSocket server"
    send_ws_connect_request(access_token)


# Send a connection request to Messenger ChatBot API WebSocket server
def send_ws_connect_request(access_token):

    # create connection request message in JSON format
    connect_request_msg = {
        'reqId': str(random.randint(0, 1000000)),
        'command': 'connect',
        'payload': {
            'stsToken': access_token
        }
    }
    web_socket_app.send(json.dumps(connect_request_msg))
    print('Sent:')
    print(json.dumps(
        connect_request_msg,
        sort_keys=True,
        indent=2, separators=(',', ':')))


# Function for Refreshing Tokens.  Auth Tokens need to be refreshed within 5 minutes for the WebSocket to persist
def send_ws_keepalive(access_token):

    # create connection request message in JSON format
    connect_request_msg = {
        'reqId': str(random.randint(0, 1000000)),
        'command': 'authenticate',
        'payload': {
            'stsToken': access_token
        }
    }
    web_socket_app.send(json.dumps(connect_request_msg))
    print('Sent:')
    print(json.dumps(
        connect_request_msg,
        sort_keys=True,
        indent=2, separators=(',', ':')))


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
confidence_level = 0.7


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
    results = model.predict([input_data], workers=1, verbose=1)[0]

    # filter out predictions below a threshold, and provide intent index
    results = [[i, r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # If there is no match result manually return no answer or do not understand
    if not return_list:
        print("no answer")
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
    print(sentence)
    import random
    currentContext = ''
    useDialog = False
    if (userID in context) and (context[userID] != ''):
        currentContext = context[userID]
        useDialog = True
    print(context)
    tagResult = classify(sentence)
    # Find JSON intent
    responseList = list(filter(lambda x: x['tag'] == tagResult[0][0], intents['intents']))
    #print(tagResult[0][0])
    if tagResult[0][0] =="goodbye":
        contextAction[userID] =\
                        {
                        "actionName": "goodbye",
                        "actionValue": "{} left chatroom now".format(userID)
                    }
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
            print("found in context set")
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

def process_message(message_json):  # Process incoming message from a joined Chatroom

    message_event = message_json['event']

    if message_event == 'chatroomPost':
        try:

            incoming_msg = message_json['post']['message']
            print('Receive text message: {}'.format(incoming_msg))
            sender = message_json['post']['sender']['email']
            print('From Sender:{}'.format(sender))
            resp_msg = '@{} {}'.format(sender,botResponse(incoming_msg, sender))
            print("Response:{}".format(resp_msg))
            post_message_to_chatroom(
                access_token, chatroom_id, resp_msg)

            global confidence_level
            if sender in contextAction:
                print(contextAction)
                actionValue = contextAction[sender]['actionValue']
                if contextAction[sender]['actionName'] == "goodbye":
                    contextAction.pop(sender)
                    post_message_to_chatroom(access_token, chatroom_id, str(actionValue))
                    remove_participant(access_token,chatroom_id,sender)     

                elif contextAction[sender]['actionName'] == "getsnapshot":
                    ricName = actionValue
                    resp_msg = '@'+sender+' '+getsnapshot(ricName)
                    print(resp_msg)
                    contextAction.pop(sender)
                    post_message_to_chatroom(
                        access_token, chatroom_id, str(resp_msg))
                elif contextAction[sender]['actionName'] == "gettimeseries":
                    ricName = actionValue
                    resp_msg = '@{} {}'.format(sender,gettimeseries(ricName))
                    print(resp_msg)
                    contextAction.pop(sender)
                    post_message_to_chatroom(
                        access_token, chatroom_id, resp_msg)
                elif contextAction[sender]['actionName'] == "changeconfidencelevel":
                    confidence_level = float(actionValue)
                    resp_msg = 'Confidence Level is now adjusted to {}'.format(actionValue)
                    print(resp_msg)
                    contextAction.pop(sender)
                    # add delay 1 sec otherwise may experience toomany request error
                    time.sleep(1)
                    post_message_to_chatroom(
                        access_token, chatroom_id, resp_msg)
            print(contextAction)
            print(context)
            print('Current Confidence Level {}'.format(confidence_level))
        except Exception as error:
            print('Post meesage to a Chatroom fail :{}'.format(error))

# =============================== Main Process ========================================

isRunning=True
# Running the tutorial
if __name__ == '__main__':

    if use_data_from_eikon:
        try:
            # Set Eikon AppKey to reqeuest data from Eikon Desktop via Eikon Data API
            ek.set_app_key(eikon_data_api_app_key)
        except ek.EikonError as ex:
            print("Eikon Initialize Error:{}".format(ex.message))

    print('Load data and model for AHS Chatbot...')
    # Load ML Model    
    loadmodel()

    # Setting Python Logging
    logging.basicConfig(
    format='%(levelname)s:%(name)s :%(message)s', level=log_level)

    print('Getting RDP Authentication Token')

    # Create and initiate RDPTokenManagement object
    rdp_token = RDPTokenManagement(bot_username, bot_password, app_key, 30)

    # Authenticate with RDP Token service
    access_token, expire_time, logged_in = authen_rdp(rdp_token)
    
    # if not auth_token:
    if not access_token:
        sys.exit(1)

    print('Successfully Authenticated ')
    # Checking if the chatroom was created by previous run
        
    print('Checking if Chatroom Name {} available'.format(bilateral_chatroom_name))
    # Get chatroom list
    status, chatroom_respone = list_chatrooms(access_token)
    chatroom_list = list(filter(lambda x:x['name'] == bilateral_chatroom_name,chatroom_respone['chatrooms']))
        
    if chatroom_list:
        # Found chatroom and get chatroom id
        chatroom_id=chatroom_list[0]['chatroomId']
    else:
        # Chatroom not found, create a new Bilateral chatroom
        print("Unable to find Bilateral Chatroom {}".format(bilateral_chatroom_name))
        print("Create new bilateral chatroom name {}".format(bilateral_chatroom_name))
        chatroom_id=create_bilateral_chatroom(access_token,bilateral_chatroom_name)
        
    #print current charoom id and invite user
    print('Chatroom Id is ', chatroom_id)
    print('\nJoining Chatroom Id:{}'.format(chatroom_id))
    join_chatroom(access_token,chatroom_id)
    for mail in recipient_email:
        print('\nInvite {} to Chatroom Id:{}'.format(mail, chatroom_id))
        inviteToChatRooms(mail, chatroom_id)
    
    #Wait 2 sec and post welcome message to chatroom.
    time.sleep(2)
    post_message_to_chatroom(access_token,chatroom_id,"Welcome to AHS Bot Chatroom. Type 'bye' to exit this Chat room")
    
    # Connect to a Chatroom via a WebSocket connection
    print('Connecting to WebSocket %s ... ' % (ws_url))
    web_socket_app = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        subprotocols=['messenger-json'])

    web_socket_app.on_open = on_open
    # Event loop
    wst = threading.Thread(
        target=web_socket_app.run_forever,
        kwargs={'sslopt': {'check_hostname': False}})
    wst.start()

    try:
        while True:
            # Give 60 seconds to obtain the new security token and send reissue
            #logging.debug('expire_time = %s' %(expire_time))
            if int(expire_time) > 60: 
                time.sleep(int(expire_time) - 60) 
            else:
                # Fail the refresh since value too small
                sys.exit(1)

            print('Refresh Token ')
            access_token, refresh_token, expire_time, logged_in = authen_rdp(rdp_token)
            if not access_token:
                sys.exit(1)
                
            # Update authentication token to the WebSocket connection.
            if logged_in:
                send_ws_keepalive(access_token)

    except KeyboardInterrupt:
        web_socket_app.close()
