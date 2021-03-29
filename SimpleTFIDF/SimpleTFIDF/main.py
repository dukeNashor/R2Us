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

    query_tf_idf = GetQueryTFIDF(dict_tf_idf, dict_df, word_dicts[1], number_of_documents)
    sorted_result = GetRanking(query_tf_idf, tfidf_vec_dict, metric = "cosine")

    TFIDFApp(doc_strings, file_names)


if __name__ == '__main__':
    main()