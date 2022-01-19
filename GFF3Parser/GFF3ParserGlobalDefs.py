
g_valid_types = set({
    "gene",
    "mRNA",
    "rRNA",
    "tRNA",
    "pseudogene",
    "pseudogenic_transcript",
    "five_prime_UTR",
    "three_prime_UTR",
    "transcript",
    "exon",
    "snRNA",
    "snoRNA",
    "miRNA",
    "ncRNA",
    "ncRNA_gene",
    "lnc_RNA",
    "antisense_RNA",
    "RNase_MRP_RNA",
    "V_gene_segment",
    "C_gene_segment",
    "scRNA",
    "D_gene_segment",
    "match",
    "RNase_P_RNA",
    "SRP_RNA",
    "mobile_genetic_element",
    "origin_of_replication",
    "CDS",
    "cDNA_match",
    "match_part",
    "EST_match",
    "operon",
    "promoter",
    "polypeptide",
    "mature_polypeptide",
    "intein",
    "primary_transcript",
    "spliced_leader_RNA",
    # list more types here
    })

# 1st order
g_gene_type = "gene"

# 2nd order
g_rna_types = set({
    "mRNA",
    "rRNA",
    "tRNA"
    })

# 3rd order
g_exon_types = set({
    "CDS",
    "exon"
    })