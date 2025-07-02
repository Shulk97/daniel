# This file is under a custom Research Usage Only (RUO) license.
# Please refer to the license file LICENSE for more details.
import json

with open("OCR/document_OCR/daniel/iam/compatible_fonts_iam.json", "r") as f:
    compatible_fonts_dict = json.load(f)

valid_fonts = compatible_fonts_dict["valid_fonts"]

hw_extended_valid_fonts = compatible_fonts_dict["hw_extended_valid_fonts"]

hw_valid_fonts = compatible_fonts_dict["hw_valid_fonts"]

printed_valid_fonts = compatible_fonts_dict["printed_valid_fonts"]
