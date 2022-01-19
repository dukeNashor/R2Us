from GFF3Parser import *

filePath = "H:/dev/R2Us/GFF3Parser/data/small_hailong.gff3"

if __name__ == "__main__":

    print("Parser start.")

    gff3File = GFF3Parser.OpenFile(filePath)

    parser = GFF3Parser()
    parser.GetLineOfFileStream(gff3File)

    items = parser.GetItems()

    # TODO: serialize the items into json..
    

    print("Parser end.")
    pass