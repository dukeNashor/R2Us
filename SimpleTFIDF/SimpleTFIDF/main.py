from SimpleTFIDF import *
from DataPreprocessor import *
from UIHelper import *


def main():

    # fetch data
    dataset_dir = FetchData(use_complete_group = False)

    # get file names
    file_names, idx_dict = GetFilesInDir(dataset_dir)

    # process files, get doc strings and group labels
    doc_strings, group_labels = ProcessFiles(file_names)

    word_dicts = list(map(DocumentProcessor, doc_strings))
    number_of_documents = len(word_dicts)
    dict_tf_idf, dict_df, tfidf_vec_dict = TFIDFProcessor(word_dicts)

    TFIDFApp(doc_strings, file_names, dict_tf_idf, dict_df, tfidf_vec_dict)


if __name__ == '__main__':
    main()