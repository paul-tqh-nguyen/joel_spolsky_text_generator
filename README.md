
# Joel Spolsky Blog Generator

This is a Joel Spolsky Blog Generator.

The generated blogs are driven by a neural network trained on [Joel Spolsky's blogs]([https://www.joelonsoftware.com/](https://www.joelonsoftware.com/)) via a character-level recurrent neural network.

This is a simple application of the ideas found [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

A live demo can be found at [https://paul-tqh-nguyen.github.io/joel_spolsky_text_generator/](https://paul-tqh-nguyen.github.io/joel_spolsky_text_generator/).

The model is a simple LSTM model that attempts to predict the next character given an input sequence of text. The layers of the architecture are:

 - Character Embedding Layer.
 - Dropout Layer.
 - LSTM Layers with dropout.
 - Fully Connected Layer.

We used [Pyppeteer](https://github.com/miyakogi/pyppeteer) to scrape the data and [PyTorch](https://pytorch.org/) to implement our model. 

Some of the Python libraries used include:
* [Pandas]([https://pandas.pydata.org/](https://pandas.pydata.org/))
* [Pytorch]([https://pytorch.org/](https://pytorch.org/))
* [asyncio]([https://docs.python.org/3/library/asyncio.html](https://docs.python.org/3/library/asyncio.html))
* [psutil]([https://psutil.readthedocs.io/](https://psutil.readthedocs.io/))

Also used were [json]([https://docs.python.org/3/library/json.html](https://docs.python.org/3/library/json.html)), [re]([https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)), [abc]([https://docs.python.org/3/library/abc.html](https://docs.python.org/3/library/abc.html)),  [tqdm]([https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)), [itertools]([https://docs.python.org/3/library/itertools.html](https://docs.python.org/3/library/itertools.html)),  [functools]([https://docs.python.org/3/library/functools.html](https://docs.python.org/3/library/functools.html)), [contextlib]([https://docs.python.org/3/library/contextlib.html](https://docs.python.org/3/library/contextlib.html)), [collections]([https://docs.python.org/3/library/collections.html](https://docs.python.org/3/library/collections.html)), [math]([https://docs.python.org/3/library/math.html](https://docs.python.org/3/library/math.html)), [os]([https://docs.python.org/3/library/os.html](https://docs.python.org/3/library/os.html)), [time]([https://docs.python.org/3/library/time.html](https://docs.python.org/3/library/time.html)), [warnings]([https://docs.python.org/3/library/warnings.html](https://docs.python.org/3/library/warnings.html)), [io]([https://docs.python.org/3/library/io.html](https://docs.python.org/3/library/io.html)), [random]([https://docs.python.org/3/library/random.html](https://docs.python.org/3/library/random.html)), and [subprocess]([https://docs.python.org/3/library/subprocess.html](https://docs.python.org/3/library/subprocess.html)), 

Any feedback is welcome! My contact information can be found [here](https://paul-tqh-nguyen.github.io/about/#contact).

