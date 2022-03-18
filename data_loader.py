import os
from typing import List
from csv import reader


class Essay(object):

    def __init__(self, origin, id, lang, set, gold_score, og_text):
        self.id = lang + id
        self.origin = origin
        self.lang = lang
        self.set = set
        self.gold_score = gold_score
        self.og_text = og_text
        # TODO add translated text for lang != "en"


def load_data(input_path: str) -> List[Essay]:
    data = []
    langs = [path for path in os.listdir(input_path) if os.path.isdir(input_path+"/"+path)]
    for lang in langs:
        files = [file for file in os.listdir(input_path+"/"+lang) if not (file.lower().__contains__("readme")) and os.path.isfile(input_path+"/"+lang+"/"+file)]
    return data
