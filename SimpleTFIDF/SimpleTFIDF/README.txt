Homework 1 for COMP5425
Author: Jiawei Sun
SID: 500409987

This is a self-contained application demonstrating the text retrieval based on TF-IDF,
with only numpy as 3rd party dependency.


## Requirements:

Python version : 3.8

1. numpy (for vectorized float arithmetics)

2. tkinter (this is included in newer release of python, nevertheless, I list it out as a requirement)

3. urllib, tarfile (these are used to fetch and extract data)

## User Instruction:

Simply run the main.py using your Python interpreter, e.g. use the script below:

--
python main.py
--


## Notices:

On first start-up, the app will download the dataset into ./data, and extract it;

The application will read all the data in the data folder, process the headers, extract the actual content

of each file, and use it as a record in constructing the database;

By default, it uses the mini-newsgroups dataset, for demonstration purpose; You could change to full dataset

by setting use_complete_group to True, in the line 9 of main.py;

In the user interface, navigate through the dataset using the tree view at left-top, Double-click to load the

content into the right-top query text box, which will be used as query.

Click the button *Start Query* to query the text,

-- key words are shown at bottom-right,

-- results will be shown at the bottom-left, in score-descending order. If you used a document existing in the

   dataset, then the first result will be itself, with score = 1.0


By default, the results shows the top-50 records which are most similar to the query.

Comments:

For run-time optimization, I used unordered dict in place of an ordered dict or dense vector, to accelerate

the deriving of TF-IDF, and a dict itself is much sparser than a dense vector, which is more memory-efficient.

I also extracted the hierachical label from the file headers, which could be used to lower the information

entropy, thus boosting the accuracy, but due to time limit, it is not fully implemented, the current accuracy

is satisfying but still could be improved if we take the class labels of the data.