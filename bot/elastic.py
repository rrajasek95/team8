from elasticsearch import Elasticsearch

def ping_es(app):
    es = Elasticsearch(app.config['ES_HEADER'])

    app.logger.info("Elasticsearch ping status: {}".format(es.ping()))

