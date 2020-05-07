#!/usr/bin/python3

"""
"""

# @todo fill in doc string

###########
# Imports #
###########

import os
import re
import itertools
import pandas as pd
from functools import reduce
from typing import List, Tuple

from gather_data import OUTPUT_CSV_FILE as RAW_CSV_FILE
from misc_utilities import *

###########
# Globals #
###########

PREPROCESSED_CSV_FILE = './processed_data.csv'

#######################
# Data Pre-processing #
#######################

CURRENCY_REPLACEMENT_MAP = {
    '¢': 'cents',
    '€': 'euro',
    '£': 'pounds',
}

def preprocess_currencies(blog_text: str) -> str:
    output_string = blog_text
    for currency_symbol, currency_replacement in CURRENCY_REPLACEMENT_MAP.items():
        matches = itertools.chain(
            re.finditer(r'^('+currency_symbol+r'[0-9]+)\b', output_string),
            re.finditer(r'\s('+currency_symbol+r'[0-9]+)\b', output_string),
            re.finditer(r'^([0-9]+'+currency_symbol+r')\b', output_string),
            re.finditer(r'\s([0-9]+'+currency_symbol+r')\b', output_string),
        )
        for match in matches:
            match_string = match.group()
            number_string = match_string.replace(currency_symbol,'')
            output_string = output_string.replace(match_string, number_string+' '+currency_replacement)
        output_string = output_string.replace(currency_symbol, currency_replacement)
    return output_string

def preprocess_special_characters(blog_text: str) -> str:
    output_string = blog_text
    output_string = output_string.replace('→', '->')
    output_string = output_string.replace('″', '"')
    output_string = output_string.replace('′', "'")
    output_string = output_string.replace('…', '...')
    output_string = output_string.replace('”', '"')
    output_string = output_string.replace('“', '"')
    output_string = output_string.replace('’', "'")
    output_string = output_string.replace('‘', "'")
    output_string = output_string.replace('—', '-')
    output_string = output_string.replace('–', '-')
    output_string = output_string.replace('½', '1/2')
    output_string = output_string.replace('»', '>>')
    output_string = output_string.replace('«', '<<')
    output_string = output_string.replace('\xa0', ' ')
    output_string = output_string.replace('×', 'x')
    output_string = output_string.replace('Å', 'A')
    output_string = output_string.replace('à', 'a')
    output_string = output_string.replace('â', 'a')
    output_string = output_string.replace('å', 'a')
    output_string = output_string.replace('ā', 'a')
    output_string = output_string.replace('ç', 'c')
    output_string = output_string.replace('è', 'e')
    output_string = output_string.replace('é', 'e')
    output_string = output_string.replace('ê', 'e')
    output_string = output_string.replace('ë', 'e')
    output_string = output_string.replace('ì', 'i')
    output_string = output_string.replace('ï', 'i')
    output_string = output_string.replace('ö', 'o')
    output_string = output_string.replace('Ø', 'o')
    output_string = output_string.replace('ñ', 'n')
    output_string = output_string.replace('ü', 'u')
    output_string = output_string.replace('©', '(c)')
    output_string = output_string.replace('®', '(R)')
    return output_string

def pervasively_replace(input_string: str, old: str, new: str) -> str:
    while old in input_string:
        input_string = input_string.replace(old, new)
    return input_string

def preprocess_white_space(blog_text: str) -> str:
    output_string = blog_text
    output_string = pervasively_replace(output_string, '  ', ' ')
    output_string = pervasively_replace(output_string, '\n\n', '\n')
    output_string = pervasively_replace(output_string, '\t\t', '\t')
    return output_string

def preprocess_blog_text(blog_text: str) -> str:
    output_string = blog_text
    output_string = preprocess_currencies(output_string)
    output_string = preprocess_special_characters(output_string)
    output_string = preprocess_white_space(output_string)
    return output_string

##########
# Driver #
##########

def preprocess_data() -> None:
    df = pd.read_csv(RAW_CSV_FILE)
    df.blog_text = df.blog_text.apply(preprocess_blog_text)
    df.to_csv(PREPROCESSED_CSV_FILE, index=False)
    return

if __name__ == '__main__':
    preprocess_data()
