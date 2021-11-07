# 11-711 Group Project

## Team Members

* Ziqi Liu (ziqil2)
* Yilin Wang (yilinwan)
* Ethan Wu (yongyiw)


## Fine-Grained Emotions

### Usage

```bash
cd goemotions
./setup.sh
./reproduce.sh
```

### Related Work

[GoEmotions: A Dataset of Fine-Grained Emotions](https://arxiv.org/abs/2005.00547)


## ~~Semantic Textual Similarity~~

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


## ~~Multi-hop Question Answering~~

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
