import csv
import time

import translator

head = ["Id", "EssaySet", "Score", "OriginalText", "OriginalLanguage", "EnglishText", "GermanText", "SpanishText"]


reader = csv.reader(open("data/en_finn.csv", "r"))
reader.__next__()
r2 = csv.reader(open("data/en_joey-.csv", "r"))
r2.__next__()
writer = csv.writer(open("data/en_joey.csv", "w"))
writer.writerow(head)

scores = {}
for row in r2:
    id = row[1]
    scores[id] = row[3]


for row in reader:
    writer.writerow((row[0], row[1], scores.get(row[0]), row[3], "EN", row[5], row[6], row[7]))
    time.sleep(0.3)



