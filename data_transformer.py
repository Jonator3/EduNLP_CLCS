import csv
import time

import translator

head = ["Id", "EssaySet", "Score", "OriginalText", "OriginalLanguage", "EnglishText", "GermanText", "SpanishText"]


reader1d = open("ASAP_übersetzt/test_public_prompt1 de.txt", "r")
reader2d = open("ASAP_übersetzt/test_public_prompt2 de.txt", "r")
reader10d = open("ASAP_übersetzt/test_public_prompt10 de.txt", "r")
reader1e = open("ASAP_übersetzt/test_public_prompt1 es.txt", "r")
reader2e = open("ASAP_übersetzt/test_public_prompt2 es.txt", "r")
reader10e = open("ASAP_übersetzt/test_public_prompt10 es.txt", "r")
reader = csv.reader(open("data/en_test.csv", "r"))
reader.__next__()
writer = csv.writer(open("data/en_test.csv", "w"))
writer.writerow(head)

data = {}
for row in reader:
    data[row[0]] = [row[0], row[1], row[2], row[3], row[4], row[5], "", ""]

for line in reader1d.readlines():
    split = line.split(" ")
    text = " ".join(split[5:])
    key = "en"+split[0]
    if data.get(key) is not None:
        data[key][6] = text
for line in reader2d.readlines():
    split = line.split(" ")
    text = " ".join(split[5:])
    key = "en"+split[0]
    if data.get(key) is not None:
        data[key][6] = text
for line in reader10d.readlines():
    split = line.split(" ")
    text = " ".join(split[5:])
    key = "en"+split[0]
    if data.get(key) is not None:
        data[key][6] = text

for line in reader1e.readlines():
    split = line.split(" ")
    text = " ".join(split[5:])
    key = "en"+split[0]
    if data.get(key) is not None:
        data[key][7] = text
for line in reader2e.readlines():
    split = line.split(" ")
    text = " ".join(split[5:])
    key = "en"+split[0]
    if data.get(key) is not None:
        data[key][7] = text
for line in reader10e.readlines():
    split = line.split(" ")
    text = " ".join(split[5:])
    key = "en"+split[0]
    if data.get(key) is not None:
        data[key][7] = text

for key in data.keys():
    row = data.get(key)
    if 10 > int(row[1]) >= 3:
        continue
    writer.writerow(row)

