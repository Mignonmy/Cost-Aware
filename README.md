# When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation

# Overview
We introduce cost-aware RAG, a setting where retrieved evidence is assigned access-cost tiers and systems must answer under an explicit evidence-access budget. We instantiate this setting by augmenting MS MARCO v2.1 with access-friction tiers and evaluate budgeted evidence selection across general-domain and domain-specific QA benchmarks.

# Set Up



# Prepare Data
## Corpus
For general corpus, we use [MSMARCOv2.1](https://trec-rag.github.io/annoucements/2024-corpus-finalization/) (Segmented version).
For domain-specific corpus, we use [Textbooks](https://huggingface.co/datasets/MedRAG/textbooks).
Put it in corpus/data folder.

## Test Dataset
For general datasets, we use [KILT](https://github.com/facebookresearch/KILT).
For domain-specific datasets, we use [MedQA-US and MMLU-Med](https://github.com/gzxiong/MIRAGE).
Put it in test_dataset folder.



# Build Corpus Index
We recommand using [RetServe](https://github.com/xhd0728/RetServe) to build embedding index. And follow the instruction in RetServe to construct the jsonl data, build embedding index and retrieve documents.
Put the retrieved result in test_dataset/retrieved_result_on_marco_textbooks folder.


# Corpus Annotation
Extract domain from url
```
python corpus/data/domain_analysis.py
```

Classify top20K domains using GPT
```
python corpus/data/corpus_division_newtiertable.py
```

Assign long tail domain based on proportion observed in top20K domains
```
python corpus/data/classification_analysis.py
```

Merge tier 2 and 3
```
python corpus/data/merge_tier_and_stats.py
```

Construct the meta data of the corpus, including docid_domain_mapping and docid_tier_mapping
```
python corpus/data/domain_docid_mapping.py
python corpus/data/generate_docid_tier_mapping.py
```
Add domain-specific corpus to tier 2
```
python corpus/data/add_new_corpus_docids_to_tier2.py --existing_mapping corpus/meta_data/docid_tier_mapping.json --new_docids path_to_textbooks --output orpus/meta_data/docid_tier_mapping_addtextbooks.json
```


# RAG Experiment

# Agent Experiment


# Contact
If you have questions, suggestions, and bug reports, please email:
```
mignonmiyoung@gmail.com
```