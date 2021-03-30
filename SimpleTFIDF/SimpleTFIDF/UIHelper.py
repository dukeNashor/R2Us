import os

from collections import OrderedDict

import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk

from SimpleTFIDF import *

g_number_of_results = 50

class TFIDFApp:

    def __init__(self, doc_strings, file_names, dict_tf_idf, dict_df, tfidf_vec_dict):

        self.doc_strings = doc_strings
        self.file_names = file_names
        self.dict_tf_idf = dict_tf_idf
        self.dict_df = dict_df
        self.tfidf_vec_dict = tfidf_vec_dict
        self.number_of_documents = len(self.file_names)
        self.labels = []

        self.root = tk.Tk()
        self.root.title('Homework 1 App by Jiawei Sun(500409987)')
        self.root.geometry("1200x1000")
        self.root.config(bg="skyblue")

        # initialize frames
        self.lt_frame = tk.Frame(self.root)
        self.lt_frame.grid(row=0, column=0, padx=10, pady=5)
        self.rt_frame = tk.Frame(self.root)
        self.rt_frame.grid(row=0, column=1, padx=10, pady=5)
        self.lb_frame = tk.Frame(self.root)
        self.lb_frame.grid(row=1, column=0, padx=10, pady=5)
        self.rb_frame = tk.Frame(self.root)
        self.rb_frame.grid(row=1, column=1, padx=10, pady=5)

        # left-top frame
        tk.Label(self.lt_frame, text = "Double click to select query", padx=10, pady=5).pack()
        self.tree_view = ttk.Treeview(self.lt_frame, height=25)
        self.tree_view.pack()
        self.FillUpTreeview(doc_strings, file_names)
        self.tree_view.bind("<Double-1>", self.OnSelectionChanged)
        self.query_button = tk.Button(self.rt_frame, text = "Start Query", command = self.UpdateQueryResult, bg='#e8b015')
        self.query_button.pack()

        # right-top frame
        tk.Label(self.rt_frame, text = "Query text\n(you can also type in text and click\n*Start Query* button to query your text)", padx=10, pady=5).pack()
        self.query_view = tk.scrolledtext.ScrolledText(self.rt_frame, height=32, width=50)
        self.query_view.pack()
    
        # left-bottom frame
        tk.Label(self.lb_frame, text = "Query result", padx=10, pady=5).pack()
        self.query_result_view = tk.scrolledtext.ScrolledText(self.lb_frame, height=27, width=100)
        self.query_result_view.pack()

        # right-bottom frame
        tk.Label(self.rb_frame, text = "key words", padx=10, pady=5).pack()
        self.word_view = tk.scrolledtext.ScrolledText(self.rb_frame, height=27, width=50)
        self.word_view.pack()


        # start event loop
        self.root.mainloop()

    
    def OnSelectionChanged(self, event):
        item = self.tree_view.selection()[0]
        idx_str = self.tree_view.item(item,"text")
        print("you clicked on", idx_str)
        if (not idx_str.isnumeric()):
            print("not a number")
            return

        idx = int(idx_str)

        self.UpdateQueryView(idx)
        self.UpdateQueryWords()
        self.UpdateQueryResult()
        return
        
    # update right-top frame
    def UpdateQueryView(self, idx):
        self.query_view.delete(1.0, "end")
        self.query_view.insert(1.0, self.doc_strings[idx])


    # update right-bottom frame
    def UpdateQueryWords(self):
        
        query_document = self.query_view.get("1.0", tk.END)
        query_words = ""
        word_dict = DocumentProcessor(query_document)

        for key, value in word_dict.items():
            query_words += key + "\t:\t" + str(value) + "\n"

        self.word_view.delete(1.0, "end")
        self.word_view.insert(1.0, query_words)


    # update left-bottom frame
    def UpdateQueryResult(self):

        query_document = self.query_view.get("1.0", tk.END)
        if (query_document == "\n"):
            print("query is empty")
            return

        # query
        query_tf_idf = GetQueryTFIDF(self.dict_tf_idf, self.dict_df, query_document, self.number_of_documents)

        # get sorted results
        sorted_result = GetRanking(query_tf_idf, self.tfidf_vec_dict, metric = "cosine")

        # create string to show:
        show_string = " Query result\n"

        for i in range(min(len(sorted_result), g_number_of_results)):
            show_string += "\n Rank "+ str(i + 1) +": ###\t" + str(sorted_result[i][0]) + "\t### label: " + self.labels[sorted_result[i][0]] + "\t### Score: " + "{:.7f}".format(sorted_result[i][1]) + "\n"

        # update to query result view
        self.query_result_view.delete(1.0, "end")
        self.query_result_view.insert(1.0, show_string)

        return
    

    # fill up the tkinker.Treeview object with list of documents
    # assume the 3 lists are of the same order.
    def FillUpTreeview(self, doc_strings, file_names):
    
        # allocate columns
        self.tree_view["columns"]=("label","file name")
        self.tree_view.column("#0", width=220, minwidth=0, stretch=tk.ON)
        self.tree_view.column("label", width=220, minwidth=150, stretch=tk.ON)
        self.tree_view.column("file name", width=100, minwidth=150)
    
        # set headings
        self.tree_view.heading("#0",text="index",anchor=tk.W)
        self.tree_view.heading("label", text="label",anchor=tk.W)
        self.tree_view.heading("file name", text="name",anchor=tk.W)

        # get folder names
        self.labels = list(map(lambda x : os.path.normpath(x).split(os.path.sep)[-2], file_names))
        folder_names = list(OrderedDict.fromkeys(self.labels))
        folder_name_dict = { n : i for i, n in enumerate(folder_names) }

        # insert folders
        folders = []
        for i in range(len(folder_names)):
            folders.append(self.tree_view.insert("", index = i, text = folder_names[i], values = ("","","")))
    
        # loop through file_names, append them to correct folder
        for i in range(len(file_names)):
            fi = folder_name_dict[self.labels[i]]
            self.tree_view.insert(folders[fi], index = "end", text = str(i), values=(self.labels[i], os.path.basename(file_names[i])))




