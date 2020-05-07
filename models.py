#!/usr/bin/python3
'#!/usr/bin/python3 -OO'

'''
'''

# @todo fill in the doc string

###########
# Imports #
###########

import os
import math
import json
import itertools
import random
import pandas as pd
from typing import List, Callable, Tuple
from functools import reduce
from abc import ABC, abstractmethod
from collections import OrderedDict

from preprocess_data import PREPROCESSED_CSV_FILE
from misc_utilities import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

###########
# Globals #
###########

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8
NUMBER_OF_PREDICTED_CHARACTERS_TO_DEMONSTRATE = 200

OUTPUT_DIRECTORY = './default_output'
INPUT_SEQUENCE_LENGTH = 100
BATCH_SIZE = 256
NUMBER_OF_EPOCHS = 100
NUMBER_OF_RELEVANT_RECENT_EPOCHS = 3
MAX_DATASET_SIZE = None

TRAINING_PORTION = 0.7
VALIDATION_PORTION = 1-TRAINING_PORTION

EMBEDDING_SIZE = 512
ENCODING_HIDDEN_SIZE = 128
NUMBER_OF_ENCODING_LAYERS = 2
DROPOUT_PROBABILITY = 0.5

#############
# Load Data #
#############

def _input_output_pairs_from_blog_text(blog_text: str, char2idx: dict, input_string_length: int) -> List[Tuple[List[int], int]]:
    pairs: List[Tuple[List[int], int]] = []
    for snippet_index in range(len(blog_text)-input_string_length):
        input_example = blog_text[snippet_index:snippet_index+input_string_length]
        input_example = [char2idx[char] for char in input_example]
        output_example = blog_text[snippet_index+input_string_length]
        output_example = char2idx[output_example]
        input_output_pair = (input_example, output_example)
        pairs.append(input_output_pair)
    return pairs

def input_output_pairs_from_blog_text(inputs) -> List[Tuple[List[int], int]]:
    blog_text, char2idx, input_string_length = inputs
    pairs = _input_output_pairs_from_blog_text(blog_text, char2idx, input_string_length)
    return pairs

def input_output_pairs_from_blog_texts(blog_texts: List[str], char2idx: dict, input_string_length: int) -> None:
    return reduce(list.__add__, eager_map(input_output_pairs_from_blog_text, zip(blog_texts, itertools.repeat(char2idx), itertools.repeat(input_string_length))))

def initialize_numericalized_blog_dataset(input_string_length: int) -> data.Dataset:
    x_data: List[str] = []
    y_data: List[str] = []
    preprocessed_data_df = pd.read_csv(PREPROCESSED_CSV_FILE)
    blog_texts = [blog_text for blog_text in preprocessed_data_df.blog_text if len(blog_text) > input_string_length]
    idx2char = sorted(reduce(set.union, [set(blog) for blog in blog_texts]))
    char2idx = {char:idx for idx, char in enumerate(idx2char)}
    input_output_pairs = input_output_pairs_from_blog_texts(blog_texts, char2idx, input_string_length)
    if MAX_DATASET_SIZE is not None:
        input_output_pairs = input_output_pairs[:MAX_DATASET_SIZE]
    x_data, y_data = zip(*input_output_pairs)
    assert len(x_data) == len(y_data)
    x_data_tensor = torch.tensor(x_data)
    y_data_tensor = torch.tensor(y_data)
    dataset = data.TensorDataset(x_data_tensor, y_data_tensor)
    dataset.input_string_length = input_string_length
    dataset.idx2char = idx2char
    dataset.char2idx = char2idx
    dataset.alphabet_size = len(idx2char)
    return dataset

##########
# Models #
##########

class LSTMNetwork(nn.Module):
    def __init__(self, alphabet_size: int, embedding_size: int, encoding_hidden_size: int, number_of_encoding_layers: int, dropout_probability: float):
        super().__init__()
        if __debug__:
            self.alphabet_size = alphabet_size
            self.output_size = alphabet_size
            self.embedding_size = embedding_size
            self.encoding_hidden_size = encoding_hidden_size
            self.number_of_encoding_layers = number_of_encoding_layers
        self.embedding_layers = nn.Sequential(OrderedDict([
            ('embedding_layer', nn.Embedding(alphabet_size, embedding_size, max_norm=1.0)),
            ('dropout_layer', nn.Dropout(dropout_probability)),
        ]))
        self.encoding_layers = nn.LSTM(embedding_size,
                                       encoding_hidden_size,
                                       num_layers=number_of_encoding_layers,
                                       bidirectional=True,
                                       batch_first=True,
                                       dropout=dropout_probability)
        self.prediction_layers = nn.Sequential(OrderedDict([
            ('dropout_layer', nn.Dropout(dropout_probability)),
            ('fully_connected_layer', nn.Linear(encoding_hidden_size*2, alphabet_size)),
        ]))
        self.to(DEVICE)
    
    def forward(self, batch: torch.tensor):
        batch_size, sequence_length = tuple(batch.shape)
        assert tuple(batch.shape) == (batch_size, sequence_length)

        embedded_batch = self.embedding_layers(batch)
        assert tuple(embedded_batch.shape) == (batch_size, sequence_length, self.embedding_size)
        
        if __debug__:
            encoded_batch, (encoding_hidden_state, encoding_cell_state) = self.encoding_layers(embedded_batch)
        else:
            encoded_batch, _ = self.encoding_layers(embedded_batch)
        assert tuple(encoded_batch.shape) == (batch_size, sequence_length, self.encoding_hidden_size*2)
        assert tuple(encoding_hidden_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)
        assert tuple(encoding_cell_state.shape) == (self.number_of_encoding_layers*2, batch_size, self.encoding_hidden_size)

        mean_batch = torch.mean(encoded_batch,dim=1)
        assert tuple(mean_batch.shape) == (batch_size, self.encoding_hidden_size*2)
        
        prediction = self.prediction_layers(mean_batch)
        assert tuple(prediction.shape) == (batch_size, self.output_size)
        
        return prediction

##############
# Predictors #
##############

class Predictor(ABC):
    def __init__(self, output_directory: str, input_sequence_length: int, number_of_epochs: int, batch_size: int, train_portion: float, validation_portion: float, **kwargs):
        super().__init__()
        self.best_valid_loss = float('inf')
        
        self.model_args = kwargs
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size

        self.dataset = initialize_numericalized_blog_dataset(input_sequence_length)
        self.train_portion = train_portion
        self.validation_portion = validation_portion
        assert math.isclose(1.0, self.validation_portion+self.train_portion)
        number_of_training_examples = round(self.train_portion*len(self.dataset))
        number_of_validation_examples = round(self.validation_portion*len(self.dataset))
        assert number_of_training_examples+number_of_validation_examples == len(self.dataset), f'The dataset has size {len(self.dataset)} while the number of training examples ({number_of_training_examples}) and the number of validation examples ({number_of_validation_examples}) sum to {number_of_training_examples+number_of_validation_examples}'
        self.training_dataset, self.validation_dataset = torch.utils.data.random_split(self.dataset, [number_of_training_examples, number_of_validation_examples])
        self.training_dataloader = data.DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)
        self.validation_dataloader = data.DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)

        self.output_directory = output_directory
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        self.initialize_model()
    
    @abstractmethod
    def initialize_model(self) -> None:
        pass
    
    @property
    def alphabet_size(self) -> int:
        return self.dataset.alphabet_size
    
    @property
    def output_size(self) -> int:
        return self.dataset.alphabet_size

    @property
    def input_sequence_length(self) -> int:
        return self.dataset.input_string_length
    
    @property
    def input_string_length(self) -> int:
        return self.dataset.input_string_length
    
    @property
    def char2idx(self) -> dict:
        return self.char2idx
    
    @property
    def idx2char(self) -> dict:
        return self.idx2char
    
    @property
    def dataset_size(self) -> int:
        return len(self.dataset)
    
    def train_one_epoch(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_accuracy = 0
        number_of_training_batches = len(self.training_dataloader)
        self.model.train()
        for text_batch, next_characters in tqdm_with_message(self.training_dataloader, post_yield_message_func = lambda index: f'Training Loss {epoch_loss/(index+1):.8f}', total=number_of_training_batches, bar_format='{l_bar}{bar:50}{r_bar}'):
            text_batch = text_batch.to(DEVICE)
            next_characters = next_characters.to(DEVICE)
            self.optimizer.zero_grad()
            predictions = self.model(text_batch)
            loss = self.loss_function(predictions, next_characters)
            accuracy = self.scores_of_discretized_values(predictions, next_characters)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_accuracy += accuracy
        epoch_loss /= number_of_training_batches
        epoch_accuracy /= number_of_training_batches
        return epoch_loss, epoch_accuracy
    
    def validate(self) -> Tuple[float, float]:
        epoch_loss = 0
        epoch_accuracy = 0
        self.model.eval()
        iterator_size = len(self.validation_dataloader)
        with torch.no_grad():
            for text_batch, next_characters in tqdm_with_message(self.validation_dataloader, post_yield_message_func = lambda index: f'Validation Loss {epoch_loss/(index+1):.8f}', total=iterator_size, bar_format='{l_bar}{bar:50}{r_bar}'):
                text_batch = text_batch.to(DEVICE)
                next_characters = next_characters.to(DEVICE)
                predictions = self.model(text_batch)
                loss = self.loss_function(predictions, next_characters)
                accuracy = self.scores_of_discretized_values(predictions, next_characters)
                epoch_loss += loss.item()
                epoch_accuracy += accuracy
        epoch_loss /= iterator_size
        epoch_accuracy /= iterator_size
        return epoch_loss, epoch_accuracy
    
    def train(self) -> None:
        self.print_hyperparameters()
        best_saved_model_location = os.path.join(self.output_directory, 'best-model.pt')
        most_recent_validation_loss_scores = [float('inf')]*NUMBER_OF_RELEVANT_RECENT_EPOCHS
        print(f'Starting training')
        for epoch_index in range(self.number_of_epochs):
            print('\n')
            print(f'Epoch {epoch_index}')
            with timer(section_name=f'Epoch {epoch_index}'):
                train_loss, train_accuracy = self.train_one_epoch()
                valid_loss, valid_accuracy = self.validate()
                print(f'\t Train Accuracy: {train_accuracy:.8f} | Train Loss: {train_loss:.8f}')
                print(f'\t  Val. Accuracy: {valid_accuracy:.8f} |  Val. Loss: {valid_loss:.8f}')
                if valid_loss < self.best_valid_loss:
                    self.best_valid_loss = valid_loss
                    self.save_parameters(best_saved_model_location)
                    self.save_current_model_information(epoch_index, train_loss, train_accuracy, valid_loss, valid_accuracy, False)
            print('\n')
            if any(valid_loss < previous_loss for previous_loss in most_recent_validation_loss_scores):
                most_recent_validation_loss_scores.pop(0)
                most_recent_validation_loss_scores.append(valid_loss)
            else:
                print()
                print(f'Validation is not better than any of the {NUMBER_OF_RELEVANT_RECENT_EPOCHS} recent epochs, so training is ending early due to apparent convergence.')
                print()
                break
        self.save_current_model_information(epoch_index, train_loss, train_accuracy, valid_loss, valid_accuracy, True)
        return
    
    def print_hyperparameters(self) -> None:
        print()
        print(f'Model hyperparameters are:')
        print(f'        number_of_epochs: {self.number_of_epochs}')
        print(f'        batch_size: {self.batch_size}')
        print(f'        train_portion: {self.train_portion}')
        print(f'        validation_portion: {self.validation_portion}')
        print(f'        dataset_size: {self.dataset_size}')
        print(f'        alphabet_size: {self.alphabet_size}')
        print(f'        output_directory: {self.output_directory}')
        print(f'        output_size: {self.output_size}')
        for model_arg_name, model_arg_value in sorted(self.model_args.items()):
            print(f'        {model_arg_name}: {model_arg_value.__name__ if hasattr(model_arg_value, "__name__") else str(model_arg_value)}')
        print()
        print(f'The model has {self.count_parameters()} trainable parameters.')
        print(f'This processes has PID {os.getpid()}.')
        print()
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def scores_of_discretized_values(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        if __debug__:
            batch_size = y.shape[0]
        assert batch_size <= self.batch_size
        assert tuple(y.shape) == (batch_size,)
        assert tuple(y_hat.shape) == (batch_size, self.output_size)
        y_hat_discretized = y_hat.argmax(dim=1)
        assert tuple(y_hat_discretized.shape) == tuple(y.shape)
        accuracy = ((y_hat_discretized == y).bool().sum() / len(y.view(-1))).item()
        return accuracy
    
    def save_current_model_information(self, epoch_index: int, train_loss: float, train_accuracy: float, valid_loss: float, valid_accuracy: float, is_final_result: bool) -> None:
        model_info_json_file_location = os.path.join(self.output_directory, 'final_model_info.json' if is_final_result else 'model_info.json')
        if not os.path.isfile('global_best_model_score.json'):
            log_current_model_as_best = True
        else:
            with open('global_best_model_score.json', 'r') as current_global_best_model_score_json_file:
                current_global_best_model_score_dict = json.load(current_global_best_model_score_json_file)
                current_global_best_model_loss: float = current_global_best_model_score_dict['best_valid_loss']
                log_current_model_as_best = current_global_best_model_loss > self.best_valid_loss
        with open(model_info_json_file_location, 'w') as model_info_json_file:
            model_info_dict = {
                'number_of_epochs': self.number_of_epochs,
                'batch_size': self.batch_size,
                'dataset_size': len(self.dataset),
                'input_sequence_length': self.input_sequence_length,
                'train_portion': self.train_portion,
                'validation_portion': self.validation_portion,
                'number_of_training_examples': len(self.training_dataset),
                'number_of_validation_examples': len(self.validation_dataset),
                'output_directory': self.output_directory,
                'number_of_parameters': self.count_parameters(), 
                'best_valid_loss': self.best_valid_loss,
                'epoch_index': epoch_index,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
            }
            for model_arg_name, model_arg_value in sorted(self.model_args.items()):
                model_info_dict[model_arg_name] = model_arg_value.__name__ if hasattr(model_arg_value, '__name__') else str(model_arg_value)
            json.dump(model_info_dict, model_info_json_file)
        return 
    
    def save_parameters(self, parameter_file_location: str) -> None:
        torch.save(self.model.state_dict(), parameter_file_location)
        return
    
    def load_parameters(self, parameter_file_location: str) -> None:
        self.model.load_state_dict(torch.load(parameter_file_location))
        return

    @classmethod
    @abstractmethod
    def init_via_check_point_directory(self, check_point_directory: str) -> None:
        pass
    
    def predict_next_character(self, input_string: str) -> str:
        self.model.eval()
        expected_string_length = len(self.dataset[0])
        padded_input_string = input_string[-expected_string_length:] if len(input_string) > expected_string_length else (expected_string_length-len(input_string))*' '+input_string
        indexed = [self.char2idx[char] for char in input_string]
        tensor = torch.LongTensor(indexed).to(DEVICE)
        assert tuple(tensor.shape) == (len(input_string),)
        tensor = tensor.view(1,-1)
        assert tuple(tensor.shape) == (1, len(input_string))
        predictions = self.model(tensor).detach()
        assert tuple(predictions.shape) == (1, self.output_size)
        prediction = predictions[0]
        assert tuple(prediction.shape) == (self.output_size,)
        next_character_index = torch.argmax(prediction)
        for _ in range(self.output_size):
            next_character_choices = F.softmax(prediction) > torch.rand(self.output_size).to(DEVICE)
            if any(next_character_choices):
                wheres = torch.where(next_character_choices)
                where = only_one(wheres)
                next_character_index = random.choice(where).item()
                break
        next_character = self.idx2char[next_character_index]
        assert len(next_character)==1
        return next_character
    
    def append_predicted_next_characters(self, input_string: str, number_of_next_characters: int = NUMBER_OF_PREDICTED_CHARACTERS_TO_DEMONSTRATE) -> str:
        output_string = input_string
        for _ in range(number_of_next_characters):
            output_string = output_string+self.predict_next_character(output_string)
        return output_string
    
    def _demonstrate_example(self, dataset: data.Dataset, example_index: int) -> None:
        input_tensor = dataset[example_index][0]
        input_string = ''.join([self.idx2char[idx.item()] for idx in input_tensor])
        new_string = self.append_predicted_next_characters(input_string, NUMBER_OF_PREDICTED_CHARACTERS_TO_DEMONSTRATE)
        print(f'Input String    {repr(input_string)}')
        print(f'Extended String {repr(new_string)}')
        return 
    
    def demonstrate_training_example(self, example_index: int) -> None:
        return self._demonstrate_example(self.training_dataset, example_index)
    
    def demonstrate_validation_example(self, example_index: int) -> None:
        return self._demonstrate_example(self.validation_dataset, example_index)

    def generate_random_string(self, number_of_next_characters: int = NUMBER_OF_PREDICTED_CHARACTERS_TO_DEMONSTRATE) -> str:
        initial_string = ''.join(random.choice(self.idx2char) for _ in range(self.input_sequence_length))
        random_string = self.append_predicted_next_characters(initial_string, NUMBER_OF_PREDICTED_CHARACTERS_TO_DEMONSTRATE)
        random_string = random_string[self.input_sequence_length:]
        return random_string

class LSTMPredictor(Predictor):
    def initialize_model(self) -> None:
        embedding_size = self.model_args['embedding_size']
        encoding_hidden_size = self.model_args['encoding_hidden_size']
        number_of_encoding_layers = self.model_args['number_of_encoding_layers']
        dropout_probability = self.model_args['dropout_probability']
        self.model = LSTMNetwork(self.alphabet_size, embedding_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_function = nn.CrossEntropyLoss()
        return

    @classmethod
    def init_via_check_point_directory(cls, check_point_directory: str, output_directory) -> Predictor:
        pt_file_location = os.path.join(check_point_directory, 'best-model.pt')
        model_info_json_file_location = os.path.join(check_point_directory, 'model_info.json')
        assert os.path.isfile(pt_file_location)
        assert os.path.isfile(model_info_json_file_location)
        with open(model_info_json_file_location) as model_info_file_handle:
            hyperparameter_specification = json.load(model_info_file_handle)
            input_sequence_length = int(hyperparameter_specification['input_sequence_length'])
            number_of_epochs = int(hyperparameter_specification['number_of_epochs'])
            batch_size = int(hyperparameter_specification['batch_size'])
            train_portion = float(hyperparameter_specification['train_portion'] )
            validation_portion = float(hyperparameter_specification['validation_portion'])
            model_args = dict()
            model_args['embedding_size'] = int(hyperparameter_specification['embedding_size'])
            model_args['encoding_hidden_size'] = int(hyperparameter_specification['encoding_hidden_size'])
            model_args['number_of_encoding_layers'] = int(hyperparameter_specification['number_of_encoding_layers'])
            model_args['dropout_probability'] = float(hyperparameter_specification['dropout_probability'])
        predictor = cls(output_directory, input_sequence_length, number_of_epochs, batch_size, train_portion, validation_portion, **model_args)
        predictor.model.load_state_dict(torch.load(pt_file_location))
        return predictor

###############
# Main Driver #
###############

def train_model() -> None:
    predictor = LSTMPredictor(OUTPUT_DIRECTORY, INPUT_SEQUENCE_LENGTH, NUMBER_OF_EPOCHS, BATCH_SIZE, TRAINING_PORTION, VALIDATION_PORTION,
                              embedding_size=EMBEDDING_SIZE,
                              encoding_hidden_size=ENCODING_HIDDEN_SIZE, 
                              number_of_encoding_layers=NUMBER_OF_ENCODING_LAYERS,
                              dropout_probability=DROPOUT_PROBABILITY)
    predictor.train()
    return

if __name__ == '__main__':
    train_model()
