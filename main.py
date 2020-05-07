#!/usr/bin/python3 -O
'#!/usr/bin/python3 -OO' # @ todo use this

'''
'''

###########
# Imports #
###########

import argparse
import random
import os
import itertools
from misc_utilities import *

#########################
# Hyperparameter Search #
#########################

def hyperparameter_search() -> None:
    from models import LSTMPredictor
    
    number_of_epochs = 100
    training_portion, validation_portion = (0.70, 0.3)
    
    batch_size_choices = [1, 32, 64, 256]
    input_sequence_length_choices = [128, 512]
    embedding_size_choices = [128, 256, 512, 1024]
    encoding_hidden_size_choices = [128, 256, 512, 1024]
    number_of_encoding_layers_choices = [1,2,4,8]
    dropout_probability_choices = [0.0, 0.25, 0.5]
    
    hyparameter_list_choices = list(itertools.product(batch_size_choices, input_sequence_length_choices, embedding_size_choices, encoding_hidden_size_choices, number_of_encoding_layers_choices, dropout_probability_choices))
    random.seed()
    random.shuffle(hyparameter_list_choices)

    for (batch_size,input_sequence_length, embedding_size, encoding_hidden_size, number_of_encoding_layers, dropout_probability) in hyparameter_list_choices:
        output_directory = f'./results/epochs_{number_of_epochs}_train_frac_{training_portion}_valid_frac_{validation_portion}_batch_size_{batch_size}_input_length_{input_sequence_length}_embed_size_{embedding_size}_encoding_size_{encoding_hidden_size}_encoding_layers_{number_of_encoding_layers}_dropout_{dropout_probability}'
        final_output_results_file = os.path.join(output_directory, 'final_model_info.json') # @todo make this a global and use it
        if os.path.isfile(final_output_results_file):
            print(f'Skipping result generation for {final_output_results_file}.')
        else:
            with safe_cuda_memory():
                predictor = LSTMPredictor(output_directory,
                                          input_sequence_length,
                                          number_of_epochs,
                                          batch_size,
                                          training_portion,
                                          validation_portion,
                                          embedding_size=embedding_size,
                                          encoding_hidden_size=encoding_hidden_size,
                                          number_of_encoding_layers=number_of_encoding_layers,
                                          dropout_probability=dropout_probability)
                predictor.train()
    return

##########
# Driver #
##########

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = 40))
    parser.add_argument('-gather-data', action='store_true', help='Scrape the blog to create an unprocessed data CSV.') # @todo test this
    parser.add_argument('-preprocess-data', action='store_true', help='Preprocess the unprocessed data CSV into a preprocessed CSV.')
    parser.add_argument('-train-model', action='store_true', help='Trains our model on our dataset. Saves model to ./default_output/best-model.pt.') # @todo use the globals
    parser.add_argument('-hyperparameter-search', action='store_true', help='Exhaustively performs -train-model over the hyperparameter space via random search. Details of the best performance are tracked in global_best_model_score.json.') # @todo use the globals
    args = parser.parse_args()
    number_of_args_specified = sum(map(int,vars(args).values()))
    if number_of_args_specified == 0:
        parser.print_help()
    elif number_of_args_specified > 1:
        print('Please specify exactly one action.')
    elif args.gather_data:
        import gather_data
        gather_data.gather_data()
    elif args.preprocess_data:
        import preprocess_data
        preprocess_data.preprocess_data()
    elif args.train_model:
        import models
        models.train_model()
    elif args.hyperparameter_search:
        hyperparameter_search()
    else:
        raise Exception('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
