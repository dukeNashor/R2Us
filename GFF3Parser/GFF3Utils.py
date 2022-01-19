import re
from GFF3ParserGlobalDefs import *

reFindParent = re.compile("Parent=([^;]+);")


def GetParentID(str):
    m = reFindParent.findall(str)
    if (len(m) == 1):
        return m[0]
    else:
        return ""


class GFF3Item:

    def __init__(self, strs):
        self.mark = strs[0]
        self.name = strs[1]
        self.type = strs[2]
        self.startIdx = strs[3]
        self.endIdx   = strs[4]
        self.attr1 = strs[5]
        self.attr2 = strs[6]
        self.attr3 = strs[7]
        self.info = strs[8]

        # relational data member
        self.parentID = GetParentID(self.info)

    def IsGene(self):
        return self.type == g_gene_type

    def IsRNA(self):
        return self.type in g_rna_types

    def IsExon(self):
        return self.type in g_exon_types



class GFF3Gene:

    def __init__(self):
        self.children = []
        self.ID = ""
    
    def Append(self, item):
        self.children.append(item)

    def SetName(self, name):
        self.ID = name



class Structure:
    def __init__(self):
        self.type = "NONE_STRUCTURE_TYPE"
        self.name = "UNNAMED"
        self.strand = "-"
        self.start = 0
        self.end = 0
        self.sub_features = []

        self.children = []



class GeneStructure(Structure):

    def __init__(self):
        super().__init__()
        self.type = "NONE_GENE_TYPE"
        


class RNAStructure(Structure):

    def __init__(self):
        super().__init__()
        self.type = "NONE_RNA_TYPE"
