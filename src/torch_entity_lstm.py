# Author: Robert Guthrie
import time

import re
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import utils_nlp

START_TAG = "<START>"
STOP_TAG = "<STOP>"


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = torch.max(vec)
    max_score_broadcast = max_score.expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):
    def __init__(self, dataset, parameters):
        super(BiLSTM_CRF, self).__init__()
        self.token_embedding_dim = parameters['token_embedding_dimension']
        self.char_embedding_dim = parameters['character_embedding_dimension']
        self.token_hidden_dim = parameters['token_lstm_hidden_state_dimension']
        self.char_hidden_dim = parameters['character_lstm_hidden_state_dimension']
        self.num_gpus = parameters['number_of_gpus']
        self.vocab_size = dataset.vocabulary_size
        self.alphabet_size = dataset.vocabulary_size

        self.tag_to_ix = dataset.label_to_index

        maximum_label_index = max(self.tag_to_ix.values())
        self.tag_to_ix[START_TAG] = maximum_label_index + 1
        # self.tag_to_ix.move_to_end(START_TAG, last=False)
        self.tag_to_ix[STOP_TAG] = maximum_label_index + 2
        self.tagset_size = len(self.tag_to_ix)

        self.token_embeddings = nn.Embedding(self.vocab_size, self.token_embedding_dim)
        # self.char_embeddings = nn.Embedding(self.alphabet_size, self.char_embedding_dim)

        # if parameters['use_character_lstm']:
        #     self.char_lstm = nn.LSTM(self.char_embedding_dim, self.char_hidden_dim)
        #
        #     # The LSTM takes word embeddings as inputs, and outputs hidden states
        #     # with dimensionality hidden_dim.
        #     self.token_lstm = nn.LSTM(self.token_embedding_dim + self.char_hidden_dim, self.token_embedding_dim)
        #
        # else:
        self.token_lstm = nn.LSTM(self.token_embedding_dim, self.token_hidden_dim // 2,
                                  num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.token_hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

        # self.token_hidden, self.char_hidden = self.init_hidden()
        self.token_hidden = self.init_hidden()

        self.define_training_procedure(parameters)

    def init_hidden(self):
        # return ((autograd.Variable(torch.randn(2, 1, self.token_hidden_dim // 2)),
        #         autograd.Variable(torch.randn(2, 1, self.token_hidden_dim // 2))),
        #         (autograd.Variable(torch.randn(1, 1, self.char_hidden_dim)),
        #          autograd.Variable(torch.randn(1, 1, self.char_hidden_dim))))
        if self.num_gpus > 0:
            return (autograd.Variable(torch.randn(2, 1, self.token_hidden_dim // 2).cuda()),
                    autograd.Variable(torch.randn(2, 1, self.token_hidden_dim // 2)).cuda())
        else:
            return (autograd.Variable(torch.randn(2, 1, self.token_hidden_dim // 2)),
                    autograd.Variable(torch.randn(2, 1, self.token_hidden_dim // 2)))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        if self.num_gpus > 0:
            init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.).cuda()
        else:
            init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # terminal_var = forward_var
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.token_embeddings(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.token_lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.token_hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags.data])
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0, self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        if self.num_gpus > 0:
            forward_var = autograd.Variable(init_vvars.cuda())
        else:
            forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = autograd.Variable(
                torch.IntTensor(self.tagset_size).zero_().cuda() if self.num_gpus > 0 else torch.IntTensor(self.tagset_size).zero_())  # holds the backpointers for this step
            viterbivars_t = autograd.Variable(
                torch.FloatTensor(self.tagset_size).zero_().cuda() if self.num_gpus > 0 else torch.FloatTensor(self.tagset_size).zero_())  # holds the viterbi variables for this step

            for i, next_tag in enumerate(range(self.tagset_size)):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                viterbivar, best_tag_id = torch.max(next_tag_var, 1)
                bptrs_t[i] = best_tag_id
                viterbivars_t[i] = viterbivar
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (viterbivars_t + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).clone()
        # terminal_var[0, self.tag_to_ix[STOP_TAG]] = -10000.
        # terminal_var[0, self.tag_to_ix[START_TAG]] = -10000.
        path_score, best_tag_id = torch.max(terminal_var, 1)
        best_tag_id = best_tag_id.view(1, )

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id.data[0]]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id.data[0]]
            best_path.append(best_tag_id.data[0])
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()

        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def define_training_procedure(self, parameters):
        if parameters['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=parameters['learning_rate'], weight_decay=1e-4)
        else:
            raise ValueError('The lr_method parameter must be sgd.')


    def load_pretrained_token_embeddings(self, dataset, parameters, token_to_vector=None):
        if parameters['token_pretrained_embedding_filepath'] == '':
            return
        # Load embeddings
        start_time = time.time()
        print('Load token embeddings... ', end='', flush=True)
        if token_to_vector == None:
            token_to_vector = utils_nlp.load_pretrained_token_embeddings(parameters)
        number_of_loaded_word_vectors = 0
        number_of_token_original_case_found = 0
        number_of_token_lowercase_found = 0
        number_of_token_digits_replaced_with_zeros_found = 0
        number_of_token_lowercase_and_digits_replaced_with_zeros_found = 0
        for token in dataset.token_to_index.keys():
            if token in token_to_vector.keys():
                self.token_embeddings.weight.data[dataset.token_to_index[token]] = torch.from_numpy(token_to_vector[token])
                number_of_token_original_case_found += 1
            elif parameters['check_for_lowercase'] and token.lower() in token_to_vector.keys():
                self.token_embeddings.weight.data[dataset.token_to_index[token]] = torch.from_numpy(token_to_vector[token.lower()])
                number_of_token_lowercase_found += 1
            elif parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0',
                                                                               token) in token_to_vector.keys():
                self.token_embeddings.weight.data[dataset.token_to_index[token]] = torch.from_numpy(token_to_vector[re.sub('\d', '0', token)])
                number_of_token_digits_replaced_with_zeros_found += 1
            elif parameters['check_for_lowercase'] and parameters['check_for_digits_replaced_with_zeros'] and re.sub(
                    '\d', '0', token.lower()) in token_to_vector.keys():
                self.token_embeddings.weight.data[dataset.token_to_index[token]] = torch.from_numpy(token_to_vector[re.sub('\d', '0', token.lower())])
                number_of_token_lowercase_and_digits_replaced_with_zeros_found += 1
            else:
                continue
            number_of_loaded_word_vectors += 1
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        print("number_of_token_original_case_found: {0}".format(number_of_token_original_case_found))
        print("number_of_token_lowercase_found: {0}".format(number_of_token_lowercase_found))
        print("number_of_token_digits_replaced_with_zeros_found: {0}".format(
            number_of_token_digits_replaced_with_zeros_found))
        print("number_of_token_lowercase_and_digits_replaced_with_zeros_found: {0}".format(
            number_of_token_lowercase_and_digits_replaced_with_zeros_found))
        print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))
        print("dataset.vocabulary_size: {0}".format(dataset.vocabulary_size))



    def load_embeddings_from_pretrained_model(self, dataset, pretraining_dataset, pretrained_embedding_weights,
                                              embedding_type='token'):
        raise NotImplementedError

    def restore_from_pretrained_model(self, parameters, dataset, token_to_vector=None):
        raise NotImplementedError