import json
import spacy
import tqdm
import random
import pickle
import os

from vocabulary import Vocabulary

from pprint import pprint


def process_data_file(path, filename, nlp,
    limit=1000):
    with open(os.path.join(path, filename)) as train_file:
        train_data = json.load(train_file)

    formatted_chats = []

    for chat in tqdm.tqdm(train_data[:limit]):
        utterances = chat['chat']
        plot_doc = None
        review_doc = None

        plot_doc = nlp(chat['documents']['plot'])
        plot_sents = [sent for sent in plot_doc.sents ]
        review_doc = nlp(chat['documents']['review'])
        review_sents = [sent for sent in review_doc.sents ]
        comments = [nlp(comment) for comment in chat['documents']['comments']]

        full_knowledge = comments + plot_sents + review_sents
        context = []
        
        for idx, utterance in enumerate(utterances):
            formatted_chat = {}
            if idx % 2 == 1:
                # Odd turn, formulate reply based on previous turn context
                formatted_chat['context'] = context + [utterances[idx - 1]]
                formatted_chat['response'] = utterance
                utt_doc = nlp(utterance)
                knowledge_similarity = [(doc, doc.similarity(utt_doc)) for doc in full_knowledge]
                most_similar_knowledge = sorted(knowledge_similarity, reverse=True, key=lambda x: x[1])[:10]
                facts = [fact[0].text for fact in most_similar_knowledge]
                random.shuffle(facts)
                formatted_chat['facts'] = facts

                context = context + [utterances[idx - 1], utterance]
                formatted_chats.append(formatted_chat)

    with open('holle/memnet_data/{}'.format(filename), 'w') as formatted_train_file:
        json.dump(formatted_chats, formatted_train_file)
    print("All chats processed")


def generate_memnet_data():
    source_data_path = os.path.join('holle',
        'main_data')

    nlp = spacy.load("en_core_web_md")

    train_path = os.path.join(source_data_path, 'train_data.json')
    dev_path = os.path.join(source_data_path, 'dev_data.json')
    test_path = os.path.join(source_data_path, 'test_data.json')

    process_data_file(source_data_path, 'train_data.json', nlp)
    process_data_file(source_data_path, 'dev_data.json', nlp, 300)
    process_data_file(source_data_path, 'test_data.json', nlp, 300)





def generate_memnet_vocabulary():
    data = []
    with open('holle/memnet_data/train_data.json', 'r') as memnet_train_file:
        data += json.load(memnet_train_file)
    with open('holle/memnet_data/dev_data.json', 'r') as memnet_dev_file:
        data += json.load(memnet_dev_file)

    with open('holle/memnet_data/test_data.json', 'r') as memnet_test_file:
        data += json.load(memnet_test_file)

    docs = []
    for row in data:
        context_lst = row['context']
        fact_lst = row['facts']
        response = row['response']

        docs += context_lst + fact_lst + [response]

    vocab = Vocabulary.from_documents(docs)

    with open('memnet_data/vocab.pkl', 'wb') as vocab_file:
        pickle.dump(vocab, vocab_file)



if __name__ == '__main__':
    generate_memnet_vocabulary()