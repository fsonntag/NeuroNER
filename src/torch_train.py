import codecs
import os

import sklearn.metrics
import torch
import torch.nn as nn
from torch import autograd

import utils_nlp
from evaluate import remap_labels


def train_step(dataset, sequence_number, model, parameters):
    model.zero_grad()
    token_indices = dataset.token_indices['train'][sequence_number]
    sentence_in = autograd.Variable(torch.LongTensor(token_indices))
    label_indices = dataset.label_indices['train'][sequence_number]
    targets = torch.LongTensor(label_indices)

    neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

    neg_log_likelihood.backward()

    if parameters['gradient_clipping_value']:
        nn.utils.clip_grad_norm(model.parameters(), parameters['gradient_clipping_value'])

    model.optimizer.step()
    transition_params_trained = model.transitions.data.numpy()
    return transition_params_trained


def prediction_step(dataset, dataset_type, model, transition_params_trained, stats_graph_folder, epoch_number,
                    parameters, dataset_filepaths):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type, epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'UTF-8')
    original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'UTF-8')

    for i in range(len(dataset.token_indices[dataset_type])):
        token_indices = dataset.token_indices[dataset_type][i]
        sentence = autograd.Variable(torch.LongTensor(token_indices))
        score, predictions = model(sentence)

        assert (len(predictions) == len(dataset.tokens[dataset_type][i]))
        output_string = ''
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_labels = dataset.labels[dataset_type][i]
        if parameters['tagging_format'] == 'bioes':
            prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
            gold_labels = utils_nlp.bioes_to_bio(gold_labels)
        for prediction, token, gold_label in zip(prediction_labels, dataset.tokens[dataset_type][i], gold_labels):
            while True:
                line = original_conll_file.readline()
                split_line = line.strip().split(' ')
                if '-DOCSTART-' in split_line[0] or len(split_line) == 0 or len(split_line[0]) == 0:
                    continue
                else:
                    token_original = split_line[0]
                    if parameters['tagging_format'] == 'bioes':
                        split_line.pop()
                    gold_label_original = split_line[-1]
                    assert (token == token_original and gold_label == gold_label_original)
                    break
            split_line.append(prediction)
            output_string += ' '.join(split_line) + '\n'
        output_file.write(output_string + '\n')

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])

    output_file.close()
    original_conll_file.close()

    if dataset_type != 'deploy':
        if parameters['main_evaluation_mode'] == 'conll':
            conll_evaluation_script = os.path.join('.', 'conlleval')
            conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepath)
            shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepath,
                                                          conll_output_filepath)
            os.system(shell_command)
            with open(conll_output_filepath, 'r') as f:
                classification_report = f.read()
                print(classification_report)
        else:
            new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = remap_labels(all_predictions, all_y_true,
                                                                                            dataset, parameters[
                                                                                                'main_evaluation_mode'])
            print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, labels=new_label_indices,
                                                        target_names=new_label_names))

    return all_predictions, all_y_true, output_filepath


def predict_labels(model, transition_params_trained, parameters, dataset, epoch_number, stats_graph_folder,
                   dataset_filepaths):
    # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step(dataset, dataset_type, model, transition_params_trained,
                                            stats_graph_folder, epoch_number, parameters, dataset_filepaths)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
    return y_pred, y_true, output_filepaths
