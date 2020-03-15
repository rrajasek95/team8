import os

from flask import (
    Blueprint, request,
    current_app
)

from .message import call_send_api, generate_echo_message

bp = Blueprint('bot', __name__)

@bp.route('/')
def hello_world():
    return 'Hello, World!'

def handle_verification():
    VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
    mode = request.args.get('hub.mode', '')
    token = request.args.get('hub.verify_token', '')
    challenge = request.args.get('hub.challenge', '')

    if mode and token:
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            return challenge, 200
        else:
            return 'Forbidden', 403
    else:
        return 'Forbidden', 403

def send_message(response, app):
    call_send_api(response, app)
    

def handle_text_message(event, app):
    sender_ps_id = event["sender"]["id"]
    app.logger.info("Received text %s" % event["message"]["text"])

    greeting = first_entity(event["message"]["nlp"], "greetings")

    message = event["message"]["text"].strip().lower()

    response = generate_echo_message(sender_ps_id, message)

    send_message(response, app)





def first_entity(nlp, name):
    return nlp and nlp.get("entities", {}).get(name, [None])[0]

def handle_message():
    content = request.json
    
    if content['object'] == "page":
        for entry in content['entry']:
            event = entry['messaging'][0]
            current_app.logger.info(event)
            current_app.logger.info("Submitting job to executor")
            handle_text_message(event, current_app)
            # current_app.config['EXECUTOR'].submit(handle_text_message, event, current_app)
        return 'EVENT_RECEIVED'
    else:
        return 'Page Not Found', 404

@bp.route('/webhook', methods=['POST', 'GET'])
def handle_webhook():
    if request.method == "GET":
        return handle_verification()
    else:
        return handle_message()