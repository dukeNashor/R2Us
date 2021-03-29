import os
import re
from functools import reduce

import urllib.request
import tarfile

# helper function to download data from web and
# extract it to ./data;
# if use_complete_group is True, then 20_newsgroups is downloaded;
# else, mini_newsgroups is downloaded.
def FetchData(use_complete_group = True):

    data_dir = "./data"

    # create data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if use_complete_group:
        dataset_url = "https://kdd.ics.uci.edu/databases/20newsgroups/20_newsgroups.tar.gz"
        dataset_dir = "./data/20_newsgroups"
        local_path = "./data/20_newsgroups.tar.gz"
    else:
        dataset_url = "https://kdd.ics.uci.edu/databases/20newsgroups/mini_newsgroups.tar.gz"
        dataset_dir = "./data/mini_newsgroups"
        local_path = "./data/mini_newsgroups.tar.gz"
    
    # if file does not exist, download it
    if (not os.path.isfile(local_path)):
        # download tar
        print("downloading dataset from " + dataset_url + " ...")
        try:
            urllib.request.urlretrieve(dataset_url, filename = local_path)
        except:
            print("failed in downloading dataset...exiting.")
            return ""

        print("downloaded to " + local_path + " ...")

    # extract data
    if (not os.path.exists(dataset_dir)):
        # extract tar
        print("extracting dataset from " + local_path + " ...")
        try:
            tar = tarfile.open(local_path, "r:gz")
            tar.extractall(path = data_dir)
            tar.close()
        except:
            print("failed in extracting tar file...exiting.")
            return ""

        print("extracted to " + dataset_dir)


    return dataset_dir


# data preprocessing specified for the newsgroup dataset used in our homework
# we get all files in the dataset directory and process each of them with the 
# presupposed data format;
# The headers do not retain a fixed order, so we use regex to find matches,
# and read files from the bottom to get the document string by number of lines.

# get all file names in directory, including subfolders;
def GetFilesInDir(dir_name):

    file_names = [os.path.join(root, name)
                  for root, dirs, files in os.walk(dir_name)
                  for name in files]

    idx_dict = { i : n for i, n in enumerate(file_names) }

    return file_names, idx_dict

# return tuples of doc - label pairs
def ProcessFiles(file_names):

    doc_strings = []
    group_labels = []

    for name in file_names:
        #doc_string, group_labels = ProcessEntry(name)
        ds, gl = ProcessEntry(name)
        doc_strings.append(ds)
        group_labels.append(gl)

    return doc_strings, group_labels

# process one file given file name;
# returns:
# doc_string: a string of the message
# group_label: a set of labels
def ProcessEntry(file_name):

    try:
        f = open(file_name, encoding="UTF-8", errors='ignore')
        lines = list(f)

        # compile regex objects ahead of use
        re_line = re.compile(r"(^Lines: )([\d]+)")
        re_ng = re.compile(r"(^Newsgroups: )([\w.]+)")

        # initialize flags
        number_of_lines = 0
        number_of_lines_found = False
        #newsgroups_found = False

        # initialize return values
        doc_string = ""
        #group_label = set()
        group_label = set(os.path.normpath(file_name).split(os.path.sep)[-2].split("."))

        # loop through lines
        for line in reversed(lines):

            # match line number header line
            m_l = re_line.match(line)
            if (not number_of_lines_found and
                (m_l is not None) and
                (len(m_l.groups()) == 2) and
                (m_l.groups()[0] + m_l.groups()[1] == line.rstrip("\n "))):
                number_of_lines = int(m_l.groups()[1])
                number_of_lines_found = True
                # get doc string
                doc_string = reduce(lambda x, y: y + x, reversed(lines[-number_of_lines:]), "")


            ## match newsgroups header line
            #m_n = re_ng.match(line)
            #if (not newsgroups_found and
            #    (m_n is not None)):

            #    group_label = set(m_n.groups()[1].split('.'))
            #    newsgroups_found = True

            #if (number_of_lines_found and newsgroups_found):
            #    break
            if number_of_lines_found:
                break

    except:
        print("Failed to process " + file_name)
        f.close()
        return "", []

    return doc_string, group_label


# calculate score of ground truth label by label depth.
# we use the portion of number of intersection of the two groups of labels
# to define a score.
# if equal --> 1.0
# if no interection --> 0.0
# in other cases, the score is defined by min(a, b),
# where a is the # of common labels divided by # of labels of first group,
# and b is # of common labels divided by # of labels of the second group.
# for example:
#           set_a                      set_b              score
#       comp.graphics               comp.graphics          1.0               
#       comp.graphics               misc.forsale           0.0           
#   comp.sys.ibm.pc.hardware    comp.sys.mac.hardware      3/5
#       rec.sport.baseball         rec.sport.hockey        2/3                                                    

def GetLabelScore(set_a, set_b):
    
    if set_a == set_b:
        return 1.0

    common = set_a & set_b

    if len(common) == 0:
        return 0.0

    return min(len(common) / len(set_a), len(common) / len(set_b))
    

