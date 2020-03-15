from collections import Counter
import string
import logging

class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):

        if token_to_idx is None:
            token_to_idx = dict()

        self._token_to_idx = token_to_idx

        self._idx_to_token = {
            idx: token
            for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1

        if add_unk:
            self.unk_index = self.add_token(unk_token)


    def add_token(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def lookup_token(self, token):
        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("Index (%d) is not in the vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)

    def get_idx2token(self):
        return self._idx_to_token

    @classmethod
    def from_documents(cls, docs, add_unk=True, skip_punctuation=False, cutoff=1):
        logging.debug("Building vocabulary. Num docs:{}".format(len(docs)))
        vocab = cls(add_unk=add_unk)

        word_counts = Counter()

        for doc in docs:
            toks = doc.split(" ")
            for tok in toks:
                is_punctuation = tok in string.punctuation
                if (is_punctuation and not skip_punctuation) or not is_punctuation:
                    word_counts[tok] += 1

        for word, count in word_counts.items():
            logging.debug("{} {}".format(word, count))
            if count >= cutoff:
                vocab.add_token(word)
        logging.debug("Vocabulary built. Size: {}".format(len(vocab)))
        return vocab

    @classmethod
    def from_dict(cls, dictionary, add_unk=True, unk_token="<UNK>"):
        vocab = cls(token_to_idx=dictionary, add_unk=add_unk, unk_token=unk_token)

        logging.debug("Vocabulary built. Size: {}".format(len(vocab)))

        return vocab