from GFF3ParserGlobalDefs import *
from GFF3Utils import *

def IsComment(line):
    return line.lstrip().startswith("##")



class GFF3Parser:
    
    def __init__(self):
        self.items = {}

    @staticmethod
    def OpenFile(filePath):
        return open(filePath, mode='r', encoding='utf-8')

    def GetLineOfFileStream(self, fs):
        for i, line in enumerate(fs):

            if (IsComment(line)):
                print(i, ": [Comment] ", line)
                continue

            item = GFF3Parser.ProcessLine(line)
            self.items.setdefault(item.parentID, GFF3Gene())
            self.items[item.parentID].SetName(item.parentID)
            self.items[item.parentID].Append(item)
            # print(i, ": [Item] ", line)
        
        pass

    def GetItems(self):
        return self.items

    @staticmethod
    def ProcessLine(line):
        strs = line.split("\t")
        return GFF3Item(strs)

            


    