#!/usr/bin/python3 -OO

# Debugging Utilities

from contextlib import contextmanager
@contextmanager
def safe_cuda_memory():
    try:
        yield
    except RuntimeError as err:
        if 'CUDA out of memory' not in str(err):
            raise
        else:
            print("CUDA ran out of memory.")

import io
from contextlib import contextmanager
@contextmanager
def std_out(stream: io.TextIOWrapper) -> None:
    import sys
    original_std_out = sys.stdout
    sys.stdout = stream
    yield
    sys.stdout = original_std_out
    return

from contextlib import contextmanager
@contextmanager
def suppressed_output() -> None:
    import sys
    with open(os.devnull, 'w') as dev_null:
        with std_out(dev_null):
            yield
    return

from typing import Callable, Union
from contextlib import contextmanager
@contextmanager
def redirected_output(exitCallback: Union[None, Callable[[str], None]] = None) -> None:
    import sys
    from io import StringIO
    temporary_std_out = StringIO()
    with std_out(temporary_std_out):
        yield
    printed_output: str = temporary_std_out.getvalue()
    if exitCallback is not None:
        exitCallback(printed_output)
    return

def shell(shell_command: str) -> str:
    import subprocess
    return subprocess.check_output(shell_command, shell=True).decode('utf-8')

def pid() -> int:
    return os.getpid()

def file(obj) -> str:
    import inspect
    try:
        file_location = inspect.getfile(obj)
        source_file_location = inspect.getsourcefile(obj)
    except TypeError as err:
        module = inspect.getmodule(obj)
        file_location = inspect.getfile(module)
        source_file_location = inspect.getsourcefile(module)
    if file_location != source_file_location:
        print("Consider using inspect.getsourcefile as well.")
    return file_location

def source(obj) -> None:
    import inspect
    try:
        source_code = inspect.getsource(obj)
    except TypeError as err:
        obj_type = type(obj)
        source_code = inspect.getsource(obj_type)
    print(source_code)
    return

def module(obj):
    return getmodule(obj)

def doc(obj) -> None:
    import inspect
    print(inspect.getdoc(obj))
    return

from typing import Iterable
def p1(iterable: Iterable) -> None:
    for e in iterable:
        print(e)
    return

from typing import List
def current_tensors() -> List:
    import torch
    import gc
    return [e for e in gc.get_objects() if isinstance(e, torch.Tensor)]

from tqdm import tqdm
def _dummy_tqdm_message_func(index: int):
    return ''
def tqdm_with_message(iterable,
                      pre_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      post_yield_message_func: Callable[[int], str] = _dummy_tqdm_message_func,
                      *args, **kwargs):
    progress_bar_iterator = tqdm(iterable, *args, **kwargs)
    for index, element in enumerate(progress_bar_iterator):
        if pre_yield_message_func != _dummy_tqdm_message_func:
            pre_yield_message = pre_yield_message_func(index)
            progress_bar_iterator.set_description(pre_yield_message)
            progress_bar_iterator.refresh()
        yield element
        if post_yield_message_func != _dummy_tqdm_message_func:
            post_yield_message = post_yield_message_func(index)
            progress_bar_iterator.set_description(post_yield_message)
            progress_bar_iterator.refresh()

from typing import Callable
def debug_on_error(func: Callable) -> Callable:
    import pdb
    import traceback
    import sys
    def decorating_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as err:
            print(f'Exception Class: {type(err)}')
            print(f'Exception Args: {err.args}')
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    return decorating_function

TRACE_INDENT_LEVEL = 0
TRACE_INDENTATION = '    '
TRACE_VALUE_SIZE_LIMIT = 200
from typing import Callable
def trace(func: Callable) -> Callable:
    from inspect import signature
    import sys
    import random
    def human_readable_value(value) -> str:
        readable_value = repr(value)
        if len(readable_value) > TRACE_VALUE_SIZE_LIMIT:
            readable_value = readable_value[:TRACE_VALUE_SIZE_LIMIT]+'...'
        return readable_value
    def decorating_function(*args, **kwargs):
        arg_values_string = ', '.join((f'{param_name}={human_readable_value(value)}' for param_name, value in signature(func).bind(*args, **kwargs).arguments.items()))
        probably_unique_id = random.randint(10,99)
        global TRACE_INDENT_LEVEL, TRACE_INDENTATION
        entry_line = f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] {func.__name__}({arg_values_string})'
        with std_out(sys.__stdout__):
            print()
            print(entry_line)
            print()
        TRACE_INDENT_LEVEL += 1
        result = func(*args, **kwargs)
        TRACE_INDENT_LEVEL -= 1
        with std_out(sys.__stdout__):
            print()
            print(f' {TRACE_INDENTATION * TRACE_INDENT_LEVEL}[{TRACE_INDENT_LEVEL}:{probably_unique_id}] returned {human_readable_value(result)}')
            print()
        return result
    return decorating_function

BOGUS_TOKEN = object()
def dpn(expression_string: str, given_frame=None):
    """dpn == debug print name"""
    import sys
    import inspect
    try:
        frame = inspect.currentframe() if given_frame is None else given_frame
        prev_frame = frame.f_back
        macro_caller_locals = prev_frame.f_locals
        macro_caller_globals = prev_frame.f_globals
        new_var_name = f'paul_dpf_hack_{id(expression_string)}'
        new_globals = dict(macro_caller_globals)
        new_globals.update({new_var_name: BOGUS_TOKEN})
        sys.stdout = open(os.devnull, 'w')
        exec(f'{new_var_name} = {expression_string}', macro_caller_locals, new_globals)
        sys.stdout = sys.__stdout__
        var_value = new_globals[new_var_name]
        if id(var_value) == id(BOGUS_TOKEN):
            raise NameError(f"Cannot determine value of {expression_string}")
        print(f'{expression_string}: {repr(var_value)}')
    finally:
        del frame
    return var_value

class __dpf_hack_by_paul__():
    def __init__(self):
        pass
    
    def __getattr__(self, var_name):
        import inspect
        frame = inspect.currentframe()
        return dpn(var_name, frame)

dpf = __dpf_hack_by_paul__() # usage is like a='a' ; dpf.a

# Timing Utilities

from typing import Callable
from contextlib import contextmanager
@contextmanager
def timeout(time: float, functionToExecuteOnTimeout: Callable[[], None] = None) -> None:
    """NB: This cannot be nested."""
    import signal
    def _raise_timeout(*args, **kwargs):
        raise TimeoutError
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        if functionToExecuteOnTimeout is not None:
            functionToExecuteOnTimeout()
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
    return

from typing import Callable
from contextlib import contextmanager
@contextmanager
def timer(section_name: str = None, exitCallback: Callable[[], None] = None) -> None:
    import time
    start_time = time.time()
    yield
    end_time = time.time()
    elapsed_time = end_time - start_time
    if exitCallback != None:
        exitCallback(elapsed_time)
    elif section_name:
        print(f'{section_name} took {elapsed_time} seconds.')
    else:
        print(f'Execution took {elapsed_time} seconds.')
    return

# General Utilities

from contextlib import contextmanager
@contextmanager
def temp_plt_figure(*args, **kwargs) -> None:
    import matplotlib.pyplot as plt
    figure = plt.figure(*args, **kwargs)
    yield figure
    plt.close(figure)
    return

from typing import Iterable
def only_one(items: Iterable):
    assert isinstance(items, Iterable)
    assert len(items) == 1
    return items[0]

from typing import List
def at_most_one(items: List):
    return only_one(items) if items else None

from typing import List
def parallel_map(*args, **kwargs) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.map(*args, **kwargs)
    p.close()
    p.join()
    return result

from typing import List
def parallel_mapcar(func, *args) -> List:
    import multiprocessing
    p = multiprocessing.Pool()
    result = p.starmap(func, zip(*args))
    p.close()
    p.join()
    return result

from typing import Iterable, Callable,  List
def eager_map(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

from typing import Iterable, Callable,  List
def eager_map_reduce(func: Callable, iterable: Iterable) -> List:
    return list(map(func, iterable))

from typing import Iterable, Callable, List
def eager_filter(func: Callable, iterable: Iterable) -> List:
    return list(filter(func, iterable))

def identity(input):
    return input

def implies(antecedent: bool, consequent: bool) -> bool:
    return not antecedent or consequent

UNIQUE_BOGUS_RESULT_IDENTIFIER = (lambda x: x)

from typing import Generator
def uniq(iterator: Iterable) -> Generator:
    previous = UNIQUE_BOGUS_RESULT_IDENTIFIER
    for value in iterator:
        if previous != value:
            yield value
            previous = value

from typing import Iterable 
def powerset(iterable: Iterable) -> Iterable:
    from itertools import chain, combinations
    items = list(iterable)
    number_of_items = len(items)
    subset_iterable = chain.from_iterable(combinations(items, length) for length in range(1, number_of_items+1))
    return subset_iterable

def n_choose_k(n: int, k: int):
    k = min(k, n-k)
    numerator = reduce(int.__mul__, range(n, n-k, -1), 1)
    denominator = reduce(int.__mul__, range(1, k+1), 1)
    return numerator // denominator

def false(*args, **kwargs) -> bool:
    return False

def current_timestamp_string() -> str:
    import time
    return time.strftime("%Y_%m_%d_%H_%M_%S")

from typing import Iterable 
def unzip(zipped_item: Iterable) -> Iterable:
    return zip(*zipped_item)

from collections import Counter
def histogram(iterator: Iterable) -> Counter:
    from collections import Counter
    counter = Counter()
    for element in iterator:
        counter[element]+=1
    return counter

