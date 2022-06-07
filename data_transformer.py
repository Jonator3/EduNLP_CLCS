import csv
import time

import translator

head = ["Id", "EssaySet", "Score", "OriginalText", "OriginalLanguage", "EnglishText", "GermanText", "SpanishText"]


"""
# Deutsch
reader = csv.reader(open("data/de.csv", "r"))
writer = csv.writer(open("data/de.csv", "w+"))
reader.__next__()
writer.writerow(head)
for row in reader:
    data = row[0], row[1], row[2], row[3], "de", row[5], row[3], translator.translate(row[3], "ES")
    time.sleep(0.1)
    print(row[0])
    writer.writerow(data)

"""
# Englisch
print("en")
reader = csv.reader(open("data/en_test.csv", "r"))
writer = csv.writer(open("data2/en_test.csv", "w+"))
reader.__next__()
writer.writerow(head)
for row in reader:
    data = row[0], row[1], row[2], row[3], "en", row[5], translator.translate(row[3], "DE"), translator.translate(row[3], "ES")
    writer.writerow(data)
    print(row[0])
reader = csv.reader(open("data/en_train.csvv", "r"))
writer = csv.writer(open("data2/en_train.csv", "w"))
reader.__next__()
writer.writerow(head)
for row in reader:
    data = row[0], row[1], row[2], row[3], "en", row[5], translator.translate(row[3], "DE"), translator.translate(row[3], "ES")
    writer.writerow(data)
    print(row[0])
"""

# Spanisch
reader = csv.reader(open("data/es.csv", "r"))
writer = csv.writer(open("data/es.csv", "w"))
reader.__next__()
writer.writerow(head)
for row in reader:
    data = row[0], row[1], row[2], row[3], "es", row[5], translator.translate(row[3], "DE"), row[3]
    time.sleep(0.1)
    print(row[0])
    writer.writerow(data)
"""
