# Joel Spolsky Blog Generator

This is a Joel Spolsky Blog Generator.

The generated blogs are driven by a neural network trained on [Joel Spolsky's blogs]([https://www.joelonsoftware.com/](https://www.joelonsoftware.com/)).

This is a simple application of the ideas found [here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

A live demo can be found at [https://paul-tqh-nguyen.github.io/joel_spolsky_text_generator/](https://paul-tqh-nguyen.github.io/joel_spolsky_text_generator/).

The model is a simple LSTM model that attempts to predict the next character given an input sequence of text. The layers of the architecture are:

 - Character Embedding Layer.
 - Dropout Layer.
 - LSTM Layers with dropout.
 - Fully Connected Layer.

We used [Pyppeteer](https://github.com/miyakogi/pyppeteer) to scrape the data and [PyTorch](https://pytorch.org/) to implement our model.

Any feedback is welcome! My contact information can be found [here](https://paul-tqh-nguyen.github.io/about/#contact).

NB: Using [Pyppeteer](https://github.com/miyakogi/pyppeteer) was not the best choice (I generally agree with the ideas presented [here](https://medium.com/@gajus/it-is-a-really-silly-idea-to-use-puppeteer-to-scrape-the-web-da62a9f3de7e) on this topic). I chose [Pyppeteer](https://github.com/miyakogi/pyppeteer) simply because I already had a web scraper from a previous project that utilized it and because adapting the code for this project took less than an hour. 