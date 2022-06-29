#!/usr/bin/env python
import json
import re
import GFF3ParserGlobalDefs

# gene_id = 'gene0010'
# gff_file_path = 'D:/sgs-project/sgs/data/small_hailong/small_hailong.gff3'

gene_id = 'gene-b0537'
gff_file_path = r'D:\sgs-project\sgs\data\ecoli\ecoli.gff3'


def GetParentID(str):
    reFindParent = re.compile("Parent=([^;]+)")
    m = reFindParent.findall(str)
    if (len(m) == 1):
        # print(m[0])
        return m[0]
    else:
        return ""


def GetID(str):
    reFindID = re.compile("ID=([^;]+);")
    j = reFindID.findall(str)
    if (len(j) == 1):
        # print(j[0])
        return j[0]
    else:
        return ""


def GetItems(gff_file_path):
    # generate items store all_information
    with open(gff_file_path, mode='r', encoding='utf-8') as f:
        items = {}
        for line in f:
            item = {}
            if not line.startswith('#'):
                strs = line.strip().split("\t")
                types = strs[2]
                start = strs[3]
                end = strs[4]
                strand = strs[6]
                info = strs[8]
                ID = GetID(info)
                parentID = GetParentID(info)
                item_element = [types, start, end, strand, ID, parentID]
                item[parentID] = item_element
                # print(item)
                if parentID not in items.keys():
                    items[parentID] = []
                    items[parentID].append(item)
                else:
                    items[parentID].append(item)

        return items



def GffParser(gene_id):
    gene_structure = {}

    # one parse gene
    with open(gff_file_path, mode='r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                gene_line = line.strip().split("\t")
                gene_id_indo = gene_line[8]
                ID = GetID(gene_id_indo)
                if gene_id == ID:
                    gene_structure['type'] = gene_line[2]
                    gene_structure['start'] = gene_line[3]
                    gene_structure['end'] = gene_line[4]
                    gene_structure['strand'] = gene_line[6]
                    gene_structure['name'] = gene_id
                    gene_structure['sub_feature'] = []
                    gene_structure['children'] = []


        items = GetItems(gff_file_path)
        one_parser = items[gene_id]
        for i in one_parser:
            two_parser = i[gene_id]
            two_type = two_parser[0]

            two_item = {}
            two_item['type'] = two_parser[0]
            two_item['start'] = two_parser[1]
            two_item['end'] = two_parser[2]
            two_item['strand'] = two_parser[3]
            two_item['name'] = two_parser[4]
            two_item['children'] = []
            two_item['sub_feature'] = []


            if two_type in GFF3ParserGlobalDefs.g_rna_types:
                # 存children
                three_ID = two_parser[4]
                # print(three_ID)
                three_parser = items[three_ID]
                # print(three_parser)
                gene_structure['children'].append(two_item)
                for j in three_parser:
                    three_info = j[three_ID]
                    three_item = {}
                    three_item['type'] = three_info[0]
                    three_item['start'] = three_info[1]
                    three_item['end'] = three_info[2]
                    three_item['strand'] = three_info[3]
                    three_item['name'] = three_info[4]
                    three_item['children'] = []
                    three_item['sub_feature'] = []
                    two_item['sub_feature'].append(three_item)
            else:
                # 存sub_feature
                gene_structure['sub_feature'].append(two_item)

        datas_json = json.dumps(gene_structure, indent=2, sort_keys=True, ensure_ascii=False)
        print(datas_json)


GffParser(gene_id)
