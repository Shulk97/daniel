# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import json

with open('OCR/document_OCR/daniel/READ/compatible_fonts_read.json', 'r') as f:
    compatible_fonts_dict = json.loads(f.read())

valid_fonts = compatible_fonts_dict['valid_fonts']

hw_valid_fonts = compatible_fonts_dict['hw_valid_fonts']

hw_fonts_wiki = compatible_fonts_dict['hw_fonts_wiki']