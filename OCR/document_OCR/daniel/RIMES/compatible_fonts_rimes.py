# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import json

with open('OCR/document_OCR/daniel/RIMES/compatible_fonts_rimes.json', 'r') as f:
    compatible_fonts_dict = json.load(f)

valid_printed_fonts = compatible_fonts_dict['valid_printed_fonts']

valid_hw_extended_fonts = compatible_fonts_dict['valid_hw_extended_fonts']

valid_fonts = compatible_fonts_dict['valid_fonts']

printed_valid_fonts = compatible_fonts_dict['printed_valid_fonts']