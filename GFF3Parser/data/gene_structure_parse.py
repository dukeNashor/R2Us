#!/usr/bin/env python
# 注意：可以理解为 children 存放 mRNA ; sub_features 存放 exon\intron 等

#
# {
#   "gene_structure": [
#     {
#       "children": [
#         {
#           "children": [],
#           "end": 562447,
#           "name": "transcript0",
#           "start": 0,
#           "strand": "-",
#           "sub_features": [
#             {
#               "children": [],
#               "end": 206006,
#               "name": "intron2",
#               "start": 165914,
#               "strand": "-",
#               "sub_features": [],
#               "type": "intron"
#             }
#           ],
#           "type": "mRNA"
#         }
#       ],
#       "end": 764756,
#       "name": "sun",
#       "start": 0,
#       "strand": "-",
#       "sub_features": [],
#       "type": "gene"
#     }
#   ]}




file_path = './small_hailong/small_hailong.gff3'
gene_name = 'gene-OR4F16'
start = 13910
end = 50115


def gene_structure_info(file_path, gene_name, start, end):
    with open(file_path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                col_info = line.split('\t')
                gene_type = col_info[2]
                start_pos = int(col_info[3])
                end_pos = int(col_info[4])
                gene_id = col_info[8].split(';')[0].split('=')[1]
