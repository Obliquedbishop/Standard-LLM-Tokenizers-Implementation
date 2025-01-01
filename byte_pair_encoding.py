import nltk
from nltk.translate.gleu_score import sentence_gleu

nltk.download('brown')
nltk.download('reuters')
from nltk.corpus import brown
from nltk.corpus import reuters

import pickle


def merge_tokens(char_training_sentences, pair):
    new_token = ''.join(pair)
    for k, sentence in enumerate(char_training_sentences):
        for j, word_list in enumerate(sentence):
            if len(word_list) < 2:
                continue
            copy_word_list = word_list.copy()
            i = 0
            while i < (len(word_list) - 1):
                if word_list[i] == pair[0] and word_list[i+1] == pair[1]:
                    rem_chars = len(word_list) - i - 1
                    new_index = len(copy_word_list) - rem_chars - 1
                    copy_word_list[new_index] = new_token
                    copy_word_list.pop(new_index+1)
                    i += 2
                else:
                    i += 1
            char_training_sentences[k][j] = copy_word_list
    return char_training_sentences


class BPE:
    def __init__(self, vocab_size):
        self._corpus = None
        self.vocab_size = vocab_size
        self._merges = []
        self._sentence_start_token = '<s>'
        self._sentence_end_token = '</s>'
        self.vocab = {self._sentence_start_token, self._sentence_end_token}

    def fit(self, corpus):
        self._corpus = corpus
        training_sentences = self._corpus.sents()  # Each sentence is a list of word tokens.
        char_training_sentences = self._transform_word_to_char_level(training_sentences)
        self._train(char_training_sentences)
        with open('model_dump/bpe_merges.pkl', 'wb') as f:
            pickle.dump(self._merges, f)
        print(self.vocab)
    
    def transform(self, sentences):
        _merges = None
        with open('model_dump/bpe_merges.pkl', 'rb') as f:
            _merges = pickle.load(f)
        if not _merges:
            raise ValueError("Model not trained yet. Please train the model first.")
        char_sentences = self._transform_word_to_char_level(sentences)
        for pair in _merges:
            char_sentences = merge_tokens(char_sentences, pair)
        return char_sentences

    def _transform_word_to_char_level(self, training_sentences):
        char_vocab = set()
        char_training_sentences = []
        for sentence in training_sentences:
            char_sentence = []
            for i, word in enumerate(sentence):
                char_list = list(word)
                char_vocab = char_vocab.union(set(char_list))
                if i == 0:
                    char_list.insert(0, self._sentence_start_token)
                elif i == len(sentence) - 1:
                    char_list.append(self._sentence_end_token)
                char_sentence.append(char_list)
            char_training_sentences.append(char_sentence)
        self.vocab = self.vocab.union(char_vocab)
        return char_training_sentences

    def _train(self, char_training_sentences):
        n_merges = self.vocab_size - len(self.vocab)
        print("No. of merges:", n_merges)
        if n_merges <= 0:
            return
        while n_merges > 0:
            pair_freq_map = dict()
            for sentence in char_training_sentences:
                for word_list in sentence:
                    for i in range(len(word_list) - 1):
                        pair = (word_list[i], word_list[i+1])
                        if pair in pair_freq_map:
                            pair_freq_map[pair] += 1
                        else:
                            pair_freq_map[pair] = 1
            max_freq_pair = max(pair_freq_map, key=pair_freq_map.get)
            merge_tokens(char_training_sentences, max_freq_pair)
            self._merges.append(max_freq_pair)
            new_token = ''.join(max_freq_pair)
            self.vocab.add(new_token)
            n_merges -= 1
    

if __name__ == '__main__':
    bpe = BPE(vocab_size=200)
    bpe.fit(brown)
    test_sentences = reuters.sents()
    transformed_test_sentences = bpe.transform(test_sentences)
    with open('bpe_reuters_transformed.txt', 'w') as f:
        for sentence in transformed_test_sentences:
            for i, word in enumerate(sentence):
                f.write(' '.join(word) + ' | ')
            f.write('\n')





