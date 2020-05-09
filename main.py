#!/usr/bin/python3 -OO

'''
Thie file contains functionality for creating a blog-text generator driven by the blogs found at https://www.joelonsoftware.com/.

Sections:
* Imports
* Globals
* Hyperparameter Search
* Generate Random Text
* Driver
'''

###########
# Imports #
###########

import argparse
import os
import random
import math
import json
import itertools

from misc_utilities import *

###########
# Globals #
###########

RANDOMLY_GENERATED_TEXTS_OUTPUT_DIR = './randomly_generated_texts'
RANDOM_TEXT_LENGTH = 10_000
NUMBER_OF_RANDOM_TEXTS_TO_GENERATE = 3600*24
NUMBER_OF_RANDOM_TEXTS_GENERATION_BATCH_SIZE = 256

#########################
# Hyperparameter Search #
#########################

def hyperparameter_search() -> None:
    import models
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
        final_output_results_file = os.path.join(output_directory, models.FINAL_MODEL_INFO_JSON_NAME)
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

########################
# Generate Random Text #
########################

def random_text_generator(check_point_directory: str) -> Generator[str, None, None]:
    from models import LSTMPredictor
    predictor = LSTMPredictor.init_via_check_point_directory(check_point_directory, '/tmp/null/')
    number_of_full_batches, remainder = divmod(NUMBER_OF_RANDOM_TEXTS_TO_GENERATE, NUMBER_OF_RANDOM_TEXTS_GENERATION_BATCH_SIZE)
    for _ in range(number_of_full_batches):
        random_strings = predictor.generate_random_strings(NUMBER_OF_RANDOM_TEXTS_GENERATION_BATCH_SIZE, RANDOM_TEXT_LENGTH)
        for random_string in random_strings:
            yield random_string
    if remainder != 0:
        random_strings = predictor.generate_random_strings(remainder, RANDOM_TEXT_LENGTH)
        for random_string in random_strings:
            yield random_string
    os.rmdir('/tmp/null/')

def generate_json_files_of_random_text(check_point_directory: str) -> None:
    output_directory = RANDOMLY_GENERATED_TEXTS_OUTPUT_DIR
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    random_texts = random_text_generator(check_point_directory)
    for json_file_index, random_string in enumerate(random_texts):
        assert isinstance(random_string, str)
        json_file_location = os.path.join(output_directory, f'random_string_{json_file_index}.json')
        with open(json_file_location, 'w') as json_file_handle:
            json.dump({'random_text': random_string}, json_file_handle)
        print(f"Finished {json_file_location}")
    return 

##########
# Driver #
##########

@debug_on_error
def main() -> None:
    from models import OUTPUT_DIRECTORY, BEST_MODEL_PT_NAME, GLOBAL_BEST_MODEL_SCORE_JSON_NAME
    parser = argparse.ArgumentParser(formatter_class = lambda prog: argparse.HelpFormatter(prog, max_help_position = float('inf')))
    parser.add_argument('-gather-data', action='store_true', help='Scrape the blog to create an unprocessed data CSV.')
    parser.add_argument('-preprocess-data', action='store_true', help='Preprocess the unprocessed data CSV into a preprocessed CSV.')
    parser.add_argument('-train-model', action='store_true', help=f'Trains our model on our dataset. Saves model to {os.path.join(OUTPUT_DIRECTORY, BEST_MODEL_PT_NAME)}.')
    parser.add_argument('-hyperparameter-search', action='store_true', help=f'Exhaustively performs -train-model over the hyperparameter space via random search. Details of the best performance are tracked in {GLOBAL_BEST_MODEL_SCORE_JSON_NAME}.')
    parser.add_argument('-generate-random-text-from-checkpoint', dest="check_point_dir", help=f'Loads the model at the specified checkpoint directory and saves them to .json files in {RANDOMLY_GENERATED_TEXTS_OUTPUT_DIR}.')
    args = parser.parse_args()
    number_of_args_specified = sum(map(bool,vars(args).values()))
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
    elif args.check_point_dir:
        generate_json_files_of_random_text(args.check_point_dir)
    else:
        raise Exception('Unexpected args received.')
    return

if __name__ == '__main__':
    main()
