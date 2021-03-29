import os

from collections import OrderedDict

import tkinter as tk
from tkinter import ttk


class TFIDFApp:

    def __init__(self, doc_strings, file_names):

        self.doc_strings = doc_strings
        self.file_names = file_names

        self.root = tk.Tk()
        self.root.title('Homework 1 App by Jiawei Sun(500409987)')
        self.root.geometry("1200x900")
        self.root.maxsize(1200, 900)
        self.root.config(bg="skyblue")

        # initialize frames
        self.lt_frame = tk.Frame(self.root, width=400, height=600)
        self.lt_frame.grid(row=0, column=0, padx=10, pady=5)
        self.rt_frame = tk.Frame(self.root, width=500, height=300)
        self.rt_frame.grid(row=0, column=1, padx=10, pady=5)
        self.lb_frame = tk.Frame(self.root, width=800, height=400)
        self.lb_frame.grid(row=1, column=0, padx=10, pady=5)

        # left-top frame
        tk.Label(self.lt_frame, text = "Double click to select query", padx=10, pady=5).pack()
        self.tree_view = ttk.Treeview(self.lt_frame, height=25)
        self.tree_view.pack()
        self.FillUpTreeview(doc_strings, file_names)
        self.tree_view.bind("<Double-1>", self.OnSelectionChanged)


        # right-top frame
        tk.Label(self.rt_frame, text = "Query text", padx=10, pady=5).pack()
        self.query_view = tk.Text(self.rt_frame, height=30, width=50)
        self.query_view.pack()
    
        # left-bottom frame
        tk.Label(self.lb_frame, text = "Query result", padx=10, pady=5).pack()
        self.query_result_view = tk.Text(self.lb_frame, height=40, width=100)
        self.query_result_view.pack()

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
        self.UpdateQueryResult(idx)
        return
        
    # update right-top frame
    def UpdateQueryView(self, idx):
        self.query_view.delete(1.0, "end")
        self.query_view.insert(1.0, self.doc_strings[idx])

    # update left-bottom frame
    def UpdateQueryResult(self, idx):


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
        labels = list(map(lambda x : os.path.normpath(x).split(os.path.sep)[-2], file_names))
        folder_names = list(OrderedDict.fromkeys(labels))
        folder_name_dict = { n : i for i, n in enumerate(folder_names) }

        # insert folders
        folders = []
        for i in range(len(folder_names)):
            folders.append(self.tree_view.insert("", index = i, text = folder_names[i], values = ("","","")))
    
        # loop through file_names, append them to correct folder
        for i in range(len(file_names)):
            fi = folder_name_dict[labels[i]]
            self.tree_view.insert(folders[fi], index = "end", text = str(i), values=(labels[i], os.path.basename(file_names[i])))




