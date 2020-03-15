import requests
import json


def generate_echo_message(sender_ps_id, message):
    return {
        "recipient": {
            "id": sender_ps_id
        },
        "message": {
            "text": message
        }
    }


def call_send_api(request, app):

    app.logger.info("POST to URL : %s" % app.config['GRAPH_MESSAGES_ENDPOINT'])
    app.logger.info("Payload: %s" % json.dumps(request))

    r = requests.post(app.config['GRAPH_MESSAGES_ENDPOINT'], json=request)
    app.logger.info(r.status_code)
    app.logger.info(r.text)