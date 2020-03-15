from flask import Flask
from flask_executor import Executor
import os

from .bot import bp
from .elastic import ping_es

def create_app():
    app = Flask(__name__)

    app.register_blueprint(bp)

    app.config['EXECUTOR'] = Executor(app)
    app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True
    app.config['GRAPH_MESSAGES_ENDPOINT'] = "https://graph.facebook.com/v6.0/me/messages?access_token=%s" % os.getenv('PAGE_ACCESS_TOKEN', '')
    app.config['ES_HEADER'] = [{
        "host": os.getenv('ELASTIC_HOST', ''),
        "port": os.getenv('ELASTIC_PORT', ''),
    }]
    print(app.config['ES_HEADER'])
    ping_es(app)

    return app