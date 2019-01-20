import os.path

def read_text_data(filename, data_type):
    """""
    :param filename: the name of the file you want to read data from
    :param data_type: there are three data types used in this project (words,lines,sentences). Where should the the file be located in the data directory.
    :return: a dict {date_id : list of words that correspond to the data_id)
    """
    file = open(os.path.dirname(__file__) + "/../Data/" + str(data_type) + "/" + str(filename), "r")
    data_dict = {}
    for line in file.readlines():
        if "ok" in line and "#" not in line:
            lines_delimited_space = line.split(" ")
            data_line_name = lines_delimited_space[0]
            classifier_data = lines_delimited_space[-1]
            if "|" in classifier_data:
                words = classifier_data.rstrip().split("|")
            else:
                words = classifier_data.rstrip()

            data_dict[data_line_name] = words
    return data_dict

