import csv
import time

import translator


def get_dublicate_ids(file):
    import main
    duplicates = [[None]]
    data = main.load_data(file)
    data.sort(key=lambda d: d.og_text)
    for i in range(len(data) - 1):
        j = data[i]
        next = data[i + 1]
        if next.og_text.startswith(j.og_text) and len(j.og_text) * 2 > len(next.og_text):
            if duplicates[-1][-1] == j.id:
                duplicates[-1].append(next.id)
            else:
                duplicates.append([j.id, next.id])
    duplicates = duplicates[1:]
    output = []
    for set in duplicates:
        output += set[1:]
    return output


head = ["Id", "EssaySet", "Score", "OriginalText", "OriginalLanguage", "EnglishText", "GermanText", "SpanishText"]

other_lang_ids = [24,27,29,33,37,46,58,59,68,104,105,116,118,143,144,162,206,233,241,253,254,354,357,363,367,376,388,389,398,435,446,473,474,492,511,536,563,581,582,637,682,685,691,695,704,715,716,717,726,763,774,801,802,820,864,891,899,911,912,967]
trash_ids = [30,92,119,211,213,222,238,239,242,266,319,325,359,409,422,434,439,448,449,465,541,543,549,552,570,572,593,594,602,622,750,762,776,777,869,871,900,924,932,952]
duplicate_ids = get_dublicate_ids("data/en_joey-.csv")

def run(input, output):
    reader = csv.reader(open("data/"+input, "r"))
    reader.__next__()
    writer = csv.writer(open("data/"+output, "w"))
    writer.writerow(head)
    writer_t = csv.writer(open("data/"+output[:-4]+"_t.csv", "w"))
    writer_t.writerow(head)
    writer_x = csv.writer(open("data/"+"xx"+output[2:], "w"))
    writer_x.writerow(head)

    for row in reader:
        if duplicate_ids.__contains__(row[0]):
            continue
        if other_lang_ids.__contains__(int(row[0])):
            writer_x.writerow(row)
        elif trash_ids.__contains__(int(row[0])):
            writer_t.writerow(row)
        else:
            writer.writerow(row)


run("en_finn-.csv", "en_finn.csv")
run("en_joey-.csv", "en_joey.csv")

