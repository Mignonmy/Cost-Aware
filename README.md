# When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation
Source code for our paper:\
[When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation](https://arxiv.org/abs/2606.02245)

Click the link below to view our papers:

<a href='https://www.arxiv.org/2606.02245'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

If you find this work useful, please cite our paper and give us a shining star 🌟
```
@article{wu2026knowledgefreecostawareevidence,
      title={When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation}, 
      author={Mingyan Wu and Han Yang and Omer Ben-Porat and Yftah Ziser},
      year={2026},
      eprint={2606.02245},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.02245}, 
}
```

# Overview
We introduce cost-aware RAG, a setting where retrieved evidence is assigned access-cost tiers and systems must answer under an explicit evidence-access budget. We instantiate this setting by augmenting MS MARCO v2.1 with access-friction tiers and evaluate budgeted evidence selection across general-domain and domain-specific QA benchmarks.

# Set Up
**Use `git clone` to download this project**
```
git clone https://github.com/Mignonmy/Cost-Aware.git
cd Cost-Aware
```

**use the virtual environment management packages**

```
conda env create -n Cost-Aware -f cost_aware_environment.yml
```


# Prepare Data
## Corpus
For general corpus, we use [MSMARCOv2.1](https://trec-rag.github.io/annoucements/2024-corpus-finalization/) (Segmented version).
For domain-specific corpus, we use [Textbooks](https://huggingface.co/datasets/MedRAG/textbooks).
Put it in corpus/data folder.

## Test Dataset
For general datasets, we use [KILT](https://github.com/facebookresearch/KILT).
For domain-specific datasets, we use [MedQA-US and MMLU-Med](https://github.com/gzxiong/MIRAGE).
Put it in test_dataset folder.
In our experiments, we randomly sample 100 for each dataset, and we provide the sampling script in test_dataset/mini_datasets/sample_dataset_100.py , also you could find the uploaded data at test_dataset/mini_datasets.



# Build Corpus Index
We recommand using [RetServe](https://github.com/xhd0728/RetServe) to build embedding index. And follow the instruction in RetServe to construct the jsonl data, build embedding index and retrieve documents.
Put the retrieved result in test_dataset/retrieved_result_on_marco_textbooks folder. Also, you could find the uploaded data at test_dataset/retrieved_result_on_marco_textbooks.


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
python corpus/data/add_new_corpus_docids_to_tier2.py --existing_mapping corpus/meta_data/docid_tier_mapping.json --new_docids path_to_textbooks --output corpus/meta_data/docid_tier_mapping_addtextbooks.json
```
Also, you could download the constructed meta-data from [here]().


# RAG Experiment
We provide two vanilla methods (Vanilla LLM and Vanilla RAG) and five fixed evidence selection methods (relevance-only, greedy_cost_aware, knapsack, redundancy_aware, and mmr). You could find all commands used in our RAG experiments in RAG_aligned/script_command_marcotextbooks.sh and RAG_aligned/script_command_marcotextbooks_llama.sh. For more instruction about the .py file, please refer to RAG_aligned/Readme.md.


# Agent Experiment
We provide two agent methods, including per-question budgeted agent and batch-question budgeted agent. You could find all commands used in our agent experiments in in Agent/script_command_*.sh. For more instruction about the .py file, please refer to Agent/README.md.


# Contact
If you have questions, suggestions, and bug reports, please email:
```
mignonmiyoung@gmail.com
```