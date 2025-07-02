"""
Module containing the Huggingface Wikipedia class and a script for formatting its content.
"""
# MIT License

# Copyright (c) 2018 Victor Huang
# Contributors:
# Brian Davis
# Thomas Constum

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

from datasets import disable_caching, load_dataset, load_from_disk

disable_caching()
class Wikipedia:
    def __init__(self, cache_path='Datasets/raw/wiki_cache-bis', dataset_version='20220301.en'):
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        disable_caching()

        #Returns a list of text paragraphs from a random wikipedia article
        if not os.path.exists(os.path.join(cache_path,'dataset_info.json')):
            self._text_data = load_dataset("wikipedia", dataset_version, cache_dir=cache_path, beam_runner='DirectRunner')['train']
            self._text_data.save_to_disk(cache_path)
        else:
            self._text_data = load_from_disk(cache_path)
            self._text_data.shuffle()

        self.len = len(self._text_data)
        self.words=[]
        self.index=0

    def getWikiArticle(self, instance_i):
        #Returns a list of text paragraphs from a random wikipedia article
        page_content = self._text_data[instance_i]['text']

        return page_content

    def __len__(self):
        return self.len

    def __getitem__(self,i):
        article = self.getWikiArticle(i)
        return article
