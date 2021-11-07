# 11-711 Advanced NLP: Group Project

## Team Members

* Ziqi Liu ([ziqil2](https://github.com/Theorem411))
* Yilin Wang ([yilinwan](https://github.com/TonyW42))
* Ethan Wu ([yongyiw](https://github.com/yongyi-wu))


## Fine-Grained Emotions

So far, `GoEmotions` is the largest manually anotated dataset for fine-grained emotion classification. We successfully reproduced the full results of finetuning the baseline model `Cased BERT-Base` on multilable classification tasks (Section 5), each of which is based on GoEmotions taxonomy, sentiment-grouped data or Ekman's taxonomy. 

Moreover, `GoEmotions` generalizes well to other domains, as demonstrated by 3* representative transfer learning tasks (Section 6). Within the margin of error, results on `ISEAR`, `EmoInt` and `Emotion-Stimulus` datasets have been reproduced as well. 

The summary statistics for reproduced experiments are available at `goemotions/results` as individual json files. 

> *: The author conducted nine transfer learning tasks in total (Appendix H), but no code has been provided. We made significant efforts on converting data format and adjusting classification code. On the other hand, as noted by author, the chosen datasets in Section 6 already represent some degree of domain diversity. 

### Usage

```bash
cd goemotions
./setup.sh
./reproduce.sh
```

> Note: The reproduction can be completed on a single 16GB GPU. Altogether with transfer learning tasks, the experiment will take about 4 hours. 

### Related Work

* [GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)

* [An Analysis of Annotated Corpora for Emotion Classification in Text](https://aclanthology.org/C18-1179)


## ~~Semantic Textual Similarity (CANCELED)~~

Semantic Textual Similarity measures the degree to which two sentences are semantically equivalent. Although published in 2019, XLNet remains as a [champion](https://paperswithcode.com/sota/semantic-textual-similarity-on-senteval) on the Semantic Textual Similarity Benchmark (STS-B), a task from the well-known GLUE benchmark. 

Here, we reproduce the Pearson correlation score reported in the XLNet paper by finetuning the pretrained `XLNet-Large` on the SST-B dataset. The command-line output has been written to `xlnet/reproduce.txt`. 

### Usage

```bash
cd xlnet
./setup.sh
./reproduce.sh
```

> Note: To succeed in the finetuning stage, make sure to have acess to 4 GPUs with at least 16GB memory each. 

### Related Work

* [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
* [SemEval-2017 Task 1: Semantic Textual Similarity - Multilingual and Cross-lingual Focused Evaluation](https://arxiv.org/abs/1708.00055)


## ~~Multi-hop Question Answering (CANCELED)~~

We reproduce the DecompRC model's result on the HotpotQA dataset. After the successful reproduction, however, we realized that the model was no longer state-of-the-art. Also, the multi-hop QA task is intrinsically difficult. Considering time and resource constraints, we decided to abandon this project topic. With that, the command-line output during the reproduction has been saved and is available at `DecompRC/reproduce.txt`. 

### Usage

```bash
cd DecompRC
./setup.sh
./reproduce.sh
```
> Note: The reproduction is especially time- and space-consuming. Ensure to have 64GB RAM and 16 hours. Nonetheless, only 1 GPU with at least 6GB memory is needed. 

### Related Work

* [Unsupervised Question Decomposition for Question Answering](https://arxiv.org/abs/2002.09758)
* [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/pdf/1809.09600.pdf)
