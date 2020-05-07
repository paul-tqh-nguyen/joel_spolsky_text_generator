#!/usr/bin/python3

"""
"""

# @todo fill in doc string
# @todo fill in the doc string

###########
# Imports #
###########

import os
import psutil
import asyncio
import pyppeteer
import itertools
import time
import warnings
import pandas as pd
from typing import List, Tuple, Iterable, Callable, Awaitable

from misc_utilities import *

###########
# Globals #
###########

MAX_NUMBER_OF_NEW_PAGE_ATTEMPTS = 1000
NUMBER_OF_ATTEMPTS_PER_SLEEP = 1
SLEEPING_RANGE_SLEEP_TIME = 10
BROWSER_IS_HEADLESS = True

BLOG_ARCHIVE_URL = "https://www.joelonsoftware.com/archives/"

OUTPUT_CSV_FILE = './raw_data.csv'

##########################
# Web Scraping Utilities #
##########################

def _sleeping_range(upper_bound: int):
    for attempt_index in range(upper_bound):
        if attempt_index and attempt_index % NUMBER_OF_ATTEMPTS_PER_SLEEP == 0:
            time.sleep(SLEEPING_RANGE_SLEEP_TIME*(attempt_index//NUMBER_OF_ATTEMPTS_PER_SLEEP))
        yield attempt_index

EVENT_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(EVENT_LOOP)

async def _launch_browser() -> pyppeteer.browser.Browser:
    browser: pyppeteer.browser.Browser = await pyppeteer.launch({'headless': BROWSER_IS_HEADLESS,})
    return browser

BROWSER = EVENT_LOOP.run_until_complete(_launch_browser())

def scrape_function(func: Awaitable) -> Awaitable:
    async def decorating_function(*args, **kwargs):
        unique_bogus_result_identifier = object()
        result = unique_bogus_result_identifier
        global BROWSER
        for _ in _sleeping_range(MAX_NUMBER_OF_NEW_PAGE_ATTEMPTS):
            try:
                updated_kwargs = kwargs.copy()
                pages = await BROWSER.pages()
                page = pages[-1]
                updated_kwargs['page'] = page
                result = await func(*args, **updated_kwargs)
            except (pyppeteer.errors.BrowserError,
                    pyppeteer.errors.ElementHandleError,
                    pyppeteer.errors.NetworkError,
                    pyppeteer.errors.PageError,
                    pyppeteer.errors.PyppeteerError) as err:
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} {func.__name__} {err}')
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} Launching new page.')
                await BROWSER.newPage()
            except pyppeteer.errors.TimeoutError as err:
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} {func.__name__} {err}')
                warnings.warn(f'\n{time.strftime("%m/%d/%Y_%H:%M:%S")} Launching new browser.')
                browser_process = only_one([process for process in psutil.process_iter() if process.pid==BROWSER.process.pid])
                for child_process in browser_process.children(recursive=True):
                    child_process.kill()
                browser_process.kill() # @hack memory leak ; this line doesn't actually kill the process (or maybe it just doesn't the PID?)
                BROWSER = await _launch_browser()
            except Exception as err:
                raise
            if result != unique_bogus_result_identifier:
                break
        if result == unique_bogus_result_identifier:
            raise Exception
        return result
    return decorating_function

################
# Blog Scraper #
################

# Month Links

@scrape_function
async def _gather_month_links(*, page) -> List[str]:
    month_links: List[str] = []
    await page.goto(BLOG_ARCHIVE_URL)
    await page.waitForSelector("div.site-info")
    month_lis = await page.querySelectorAll('li.month')
    for month_li in tqdm_with_message(month_lis, post_yield_message_func = lambda index: f'Gathering mongth link {index}', bar_format='{l_bar}{bar:50}{r_bar}'):
        anchors = await month_li.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await page.evaluate('(anchor) => anchor.href', anchor)
        month_links.append(link)
    return month_links

def gather_month_links() -> List[str]:
    month_links = EVENT_LOOP.run_until_complete(_gather_month_links())
    return month_links

# Blog Links from Month Links

@scrape_function
async def _blog_links_from_month_link(month_link: str, *, page: pyppeteer.page.Page) -> List[str]:
    blog_links: List[str] = []
    await page.goto(month_link)
    await page.waitForSelector("div.site-info")
    blog_h1s = await page.querySelectorAll('h1.entry-title')
    for blog_h1 in blog_h1s:
        anchors = await blog_h1.querySelectorAll('a')
        anchor = only_one(anchors)
        link = await page.evaluate('(anchor) => anchor.href', anchor)
        blog_links.append(link)
    return blog_links

def blog_links_from_month_link(month_link: str) -> Iterable[str]:
    return EVENT_LOOP.run_until_complete(_blog_links_from_month_link(month_link))

def blog_links_from_month_links(month_links: Iterable[str]) -> Iterable[str]:
    month_links = tqdm_with_message(month_links, post_yield_message_func = lambda index: f'Scraping month link {index}', bar_format='{l_bar}{bar:50}{r_bar}')
    return itertools.chain(*map(blog_links_from_month_link, month_links))

# Data from Blog Links

@scrape_function
async def _data_dict_from_blog_link(blog_link: str, *, page: pyppeteer.page.Page) -> dict:
    data_dict = {'blog_link': blog_link}
    await page.goto(blog_link)
    await page.waitForSelector("div.site-info")
    articles = await page.querySelectorAll('article.post')
    article = only_one(articles)
    
    entry_title_h1s = await article.querySelectorAll('h1.entry-title')
    entry_title_h1 = only_one(entry_title_h1s)
    title = await page.evaluate('(element) => element.textContent', entry_title_h1)
    data_dict['title'] = title
    
    entry_date_divs = await article.querySelectorAll('div.entry-date')
    entry_date_div = only_one(entry_date_divs)    
    posted_on_spans = await entry_date_div.querySelectorAll('span.posted-on')
    posted_on_span = only_one(posted_on_spans)
    
    published_times = await posted_on_span.querySelectorAll('time.published')
    published_time = only_one(published_times)
    published_date = await page.evaluate('(element) => element.getAttribute("datetime")', published_time)
    data_dict['published_date'] = published_date
    updated_times = await posted_on_span.querySelectorAll('time.updated')
    updated_time = at_most_one(updated_times)
    updated_date = None
    if updated_time:
        updated_date = await page.evaluate('(element) => element.getAttribute("datetime")', updated_time)
    data_dict['updated_date'] = updated_date
    
    author_spans = await entry_date_div.querySelectorAll('span.author')
    author_span = only_one(author_spans)
    author = await page.evaluate('(element) => element.textContent', author_span)
    data_dict['author'] = author
    
    entry_meta_divs = await article.querySelectorAll('div.entry-meta')
    entry_meta_div = only_one(entry_meta_divs)
    entry_meta_div_uls = await entry_meta_div.querySelectorAll('ul.meta-list')
    entry_meta_div_ul = only_one(entry_meta_div_uls)
    entry_meta_div_ul_lis = await entry_meta_div_ul.querySelectorAll('li.meta-cat')
    entry_meta_div_ul_li = only_one(entry_meta_div_ul_lis)
    blog_tags_text = await page.evaluate('(element) => element.textContent', entry_meta_div_ul_li)
    data_dict['blog_tags'] = blog_tags_text
    
    entry_content_divs = await article.querySelectorAll('div.entry-content')
    entry_content_div = only_one(entry_content_divs)
    blog_text = await page.evaluate('(element) => element.textContent', entry_content_div)
    data_dict['blog_text'] = blog_text
    
    return data_dict

def data_dict_from_blog_link(blog_link: str) -> Iterable[dict]:
    return EVENT_LOOP.run_until_complete(_data_dict_from_blog_link(blog_link))

def data_dicts_from_blog_links(blog_links: Iterable[str]) -> Iterable[dict]:
    blog_links = blog_links if isinstance(blog_links, list) else list(blog_links)
    blog_links = tqdm_with_message(blog_links, post_yield_message_func = lambda index: f'Scraping blog link {index}', bar_format='{l_bar}{bar:50}{r_bar}')
    return map(data_dict_from_blog_link, blog_links)

###################
# Sanity Checking #
###################

def sanity_check_output_csv_file() -> None:
    import re
    import numpy as np
    df = pd.read_csv(OUTPUT_CSV_FILE)
    expected_columns = 'blog_link', 'title', 'published_date', 'updated_date', 'author', 'blog_tags', 'blog_text'
    assert all(expected_column in df.columns for expected_column in expected_columns)
    assert all(re.match(r'https://www.joelonsoftware.com/(199|200|201)[0-9]/[0-1][0-9]/[0-3][0-9]/.+/', blog_link) for blog_link in df.blog_link)
    assert all(isinstance(title,str) for title in df.title)
    assert all(re.match(r'(199|200|201)[0-9]-[0-1][0-9]-[0-9][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]\+[0-9][0-9]:[0-9][0-9]', published_date) for published_date in df.published_date)
    assert all(re.match(r'(199|200|201)[0-9]-[0-1][0-9]-[0-9][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]\+[0-9][0-9]:[0-9][0-9]', updated_date) for updated_date in df.updated_date if updated_date and updated_date==updated_date)
    assert all(df.author.unique() == np.array(['Joel Spolsky']))
    assert all(isinstance(blog_tags,str) for blog_tags in df.blog_tags.unique())
    assert all(isinstance(blog_text,str) for blog_text in df.blog_text)
    return 

##########
# Driver #
##########

def gather_data() -> None:
    with timer("Data gathering"):
        month_links = gather_month_links()
        blog_links = blog_links_from_month_links(month_links)
        rows = data_dicts_from_blog_links(blog_links)
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV_FILE, index=False)
        sanity_check_output_csv_file()
    return

if __name__ == '__main__':
    gather_data()
