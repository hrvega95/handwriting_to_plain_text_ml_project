import os.path
from contextlib import suppress
def read_text_data(filename):
    """""
    :param filename: the name of the file you want to read data from
    :return: a dict {date_id : list of words that correspond to the data_id)
    """
    file = open(os.path.dirname(__file__) + "/../Data/" + str(filename).strip(".txt") + "/" + str(filename), "r")
    data_dict = {}
    for line in file.readlines():
        if "ok" or "err" in line and "#" not in line:
            lines_delimited_space = line.split(" ")
            data_line_name = lines_delimited_space[0]
            classifier_data = lines_delimited_space[-1]
            if "|" in classifier_data:
                words = classifier_data.rstrip().split("|")
                with suppress(ValueError):
                    while True:
                        #removing periods from the data set since they usually blend with words and cannot be seperated from lines and sentences
                        words.remove(".")
            else:
                words = classifier_data.rstrip()

            data_dict[data_line_name] = words
    return data_dict


def read_word_test_data(filename):
    file = open(os.path.dirname(__file__) + "/../Data/" + str(filename).strip(".txt") + "/" + str(filename), "r")
    data_dict = {}
    for line in file.readlines():
        list_parsed = line.rstrip().split(" ")
        if list_parsed[0] != "#":
            data_dict[list_parsed[0]] = list_parsed[-1].lower()
    print(data_dict)
    return data_dict
