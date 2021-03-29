import os
import io
import re
import math

from collections import defaultdict, OrderedDict
from functools import reduce

from scipy.sparse import csr_matrix
import numpy

g_re_string = r"\W+|_|'|\""

# process a document into a word dict where keys are words in document and values are occurences of word.
def DocumentProcessor(doc_text):

    # use regular expression to get words while removing spaces, line breaks, commas, etc.
    words = list(filter(bool, re.split(g_re_string, doc_text)))
    
    words_iter = map(lambda x:x.lower(), words)
    word_dict = defaultdict(int)

    for w in words_iter:
        
        # skip stop word
        if w in g_stop_words:
            continue

        # update dict
        word_dict[w] += 1

    return word_dict


# Get TF-IDF result from given list of documents.
def TFIDFProcessor(word_dicts):

    # get unique words. use reduce to optimize performance
    unique_words = reduce(lambda x, y: (x | y.keys()), word_dicts, set())

    # initialize tf and df dict
    dict_tf = OrderedDict({ key : defaultdict(int) for key in unique_words })
    dict_df = OrderedDict({ key : 0 for key in unique_words })

    # fill up the tf and df
    for idx, dict in enumerate(word_dicts):
        #print("processing #", idx, " document..")
        for word, count in dict.items():
            #print("word:", word)
            dict_df[word] += 1
            dict_tf[word][idx] += count


    #for key in dict_tf.keys():
    #    print(key)

    # initialize tf_idf dict with shallow copy
    dict_tf_idf = dict_tf.copy()

    # calculate final tf_idf
    for word, dict in dict_tf_idf.items():
        for doc_idx, tf in dict.items():
            #print(word, "has", tf, "in doc", doc_idx)
            dict_tf_idf[word][doc_idx] = (1 + math.log(tf)) * math.log(len(word_dicts) / dict_df[word])


    tfidf_vec_dict = {}

    for idx, doc in enumerate(word_dicts):
        tfidf_vec_dict[idx] = GetQueryTFIDF(dict_tf_idf, dict_df, doc, len(word_dicts))

    return dict_tf_idf, dict_df, tfidf_vec_dict


def GetQueryTFIDF(dict_tf_idf, dict_df, query, number_of_documents):

    if type(query) is str:
        words_occurences = DocumentProcessor(query)
    if type(query) is defaultdict:
        words_occurences = query.copy() # shallow copy
    
    query_tf_idf = OrderedDict({key: 0 for key in dict_tf_idf.keys()})

    for word in query_tf_idf.keys():

        if words_occurences[word] == 0:
            continue

        query_tf_idf[word] = (1 + math.log(words_occurences[word])) * math.log(number_of_documents / dict_df[word])


    return query_tf_idf


def CalculateDistance(tfidf_1, tfidf_2, metric = "cosine", dim = 3):
    v1 = numpy.fromiter(tfidf_1.values(), dtype=float)
    v2 = numpy.fromiter(tfidf_2.values(), dtype=float)

    if metric == "cosine":
        return DistanceCosine(v1, v2)
    if metric == "euclidean":
        return DistanceEuclidean(v1, v2)
    if metric == "minkowski":
        return DistanceMinkowski(v1, v2, dimension = dim)


def GetRanking(query_tfidf, db_tfidf, metric = "cosine"):

    result = []

    for idx, entry in db_tfidf.items():
        distance = CalculateDistance(query_tfidf, entry, metric = metric)
        result.append((idx, distance))
        print("#", idx, "distance:", distance)

    # sort in descending or ascending order, based on the metric:
    return sorted(result, key = lambda x: x[1], reverse = (metric == "cosine"))

# distance metrics used in similarity calculation.
def DistanceCosine(vec1, vec2):
    return numpy.dot(vec1, vec2)/(numpy.linalg.norm(vec1)*numpy.linalg.norm(vec2))

def DistanceEuclidean(vec1, vec2):
    return numpy.sqrt(numpy.sum((vec1 - vec2)**2))

def DistanceMinkowski(vec1, vec2, dimension = 2):
    return (numpy.abs(vec1 - vec2)**dimension).sum()**(1 / dimension)

# define the common stop words
g_stop_words = {"a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't",
                "as","at","be","because","been","before","being","below","between","both","but","by","can","couldn",
                "couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during",
                "each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't",
                "having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn",
                "isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn",
                "mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or",
                "other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should",
                "should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs",
                "them","themselves","then","there","these","they","this","those","through","to","too","under","until","up",
                "ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who",
                "whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've",
                "your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm",
                "i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd",
                "we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance",
                "according","accordingly","across","act","actually","added","adj","affected","affecting","affects",
                "afterwards","ah","almost","alone","along","already","also","although","always","among","amongst",
                "announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere",
                "apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away",
                "awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings",
                "begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot",
                "can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains",
                "couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either",
                "else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone",
                "everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following",
                "follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give",
                "given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter",
                "hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie",
                "im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention",
                "inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately",
                "later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look",
                "looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile",
                "merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely",
                "nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next",
                "nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain",
                "obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside",
                "overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please",
                "plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably",
                "promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent",
                "recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted",
                "resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed",
                "seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns",
                "shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan",
                "something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying",
                "still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking",
                "tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein",
                "there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh",
                "thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try",
                "trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful",
                "usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants",
                "wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby",
                "wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose",
                "widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's",
                "ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes",
                "clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite",
                "entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates",
                "inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure",
                "t's","third","thorough","thoroughly","three","well","wonder"}