import csv
import time

import translator

head = ["Id", "EssaySet", "Score", "OriginalText", "OriginalLanguage", "EnglishText", "GermanText", "SpanishText"]


reader = csv.reader(open("data/en_finn.csv", "r"))
reader.__next__()
writer = csv.writer(open("data/en_finn2.csv", "w"))
writer.writerow(head)


for row in reader:
    writer.writerow((row[1], row[0], row[2], row[3], "EN", row[4], row[5], row[6]))



