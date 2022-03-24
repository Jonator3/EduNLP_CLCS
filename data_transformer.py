import csv
import time

import translator

head = ["Id", "EssaySet", "Score", "OriginalText", "OriginalLanguage", "EnglishText"]


# Deutsch
reader = csv.reader(open("CrossLingualData/de/germanAsap_clean.txt", "r"), delimiter="\t")
writer = csv.writer(open("data/de.csv", "w"))
reader.__next__()
writer.writerow(head)
for row in reader:
    if int(row[0]) < 633:
        continue
    data = "de"+row[0], row[1], row[2], row[4], "de", translator.translate(row[4])
    time.sleep(0.1)
    print("de", row[0])
    writer.writerow(data)

# Englisch
print("en")
reader = csv.reader(open("CrossLingualData/en/test_public.txt", "r"), delimiter="\t")
writer = csv.writer(open("data/en_test.csv", "w"))
reader.__next__()
writer.writerow(head)
for row in reader:
    data = "en"+row[0], row[1], row[2], row[4], "en", row[4]
    writer.writerow(data)
reader = csv.reader(open("CrossLingualData/en/train.tsv", "r"), delimiter="\t")
writer = csv.writer(open("data/en_train.csv", "w"))
reader.__next__()
writer.writerow(head)
for row in reader:
    data = "en"+row[0], row[1], row[2], row[4], "en", row[4]
    writer.writerow(data)

# Spanisch
reader = csv.reader(open("CrossLingualData/es/spanish_gold.txt", "r"), delimiter="\t")
writer = csv.writer(open("data/es.csv", "w"))
writer.writerow(head)
for i, row in enumerate(reader):
    data = "es"+row[0], row[1], row[2][:-2], row[3], "es", translator.translate(row[3])
    time.sleep(0.1)
    print("es", i)
    writer.writerow(data)
