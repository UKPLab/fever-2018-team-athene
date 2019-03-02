<img src="https://user-images.githubusercontent.com/29311022/27184688-27629126-51e3-11e7-9a23-276628da2430.png" height=70px/>
<img src="https://user-images.githubusercontent.com/29311022/27278631-2e19f99e-54e2-11e7-919c-f89ae0c90648.png" height=70px/>
<img src="https://user-images.githubusercontent.com/29311022/27184769-65c6583a-51e3-11e7-90e0-12a4bdf292e2.png" height=70px/>

# Multi-Sentence Textual Entailment for Claim Verification
## This repository was constructed by team Athene for the [FEVER shared task 1](http://fever.ai/2018/task.html). The system reached the third rank in the overall results and first rank on the evidence recall sub-task

This repository builds upon the baseline system repository developed by the FEVER shared task organizers: https://github.com/sheffieldnlp/fever-naacl-2018

This is an accompanying repository for our FEVER Workshop paper at EMNLP 2018. For more information see the paper: [UKP-Athene: Multi-Sentence Textual Entailment for Claim Verification](https://arxiv.org/pdf/1809.01479.pdf)

Please use the following citation:
```
@article{hanselowski2018ukp,
          title={UKP-Athene: Multi-Sentence Textual Entailment for Claim Verification},
          author={Hanselowski, Andreas and Zhang, Hao and Li, Zile and Sorokin, Daniil and Schiller, Benjamin and Schulz, Claudia and Gurevych, Iryna},
          journal={arXiv preprint arXiv:1809.01479},
          year={2018}
        }
```


Disclaimer:
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


### Requirements
* Python 3.6
* AllenNLP
* TensorFlow

### Installation 

* Download and install Anaconda (https://www.anaconda.com/)
* Create a Python Environment and activate it:
```bash 
    conda create -n fever python=3.6
    source activate fever
```
* Install the required dependencies
```bash
    pip install -r requirements.txt
```
* Download NLTK Punkt Tokenizer
```bash
    python -c "import nltk; nltk.download('punkt')"
```
* Proceed with downloading the data set, the embeddings, the models and the evidence data

### Download the FEVER data set
Download the FEVER dataset from [the website of the FEVER share task](https://sheffieldnlp.github.io/fever/data.html) into the data directory

    mkdir data
    mkdir data/fever-data
    
    #To replicate the paper, download paper_dev and paper_test files. These are concatenated for the shared task
    wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
    wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
    wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl


### Download the word embeddings

Download pretrained GloVe Vectors

    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip glove.6B.zip -d data/glove
    gzip data/glove/*.txt
    
Download pretrained Wiki FastText Vectors

    wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
    mkdir -p data/fasttext
    unzip wiki.en.zip -d data/fasttext


### Download evidence data
The data preparation consists of three steps: (1) downloading the articles from Wikipedia, (2) indexing these for the evidence retrieval and (3) performing the negative sampling for training . 

#### 1. Download Wikipedia data:

Download the pre-processed Wikipedia articles and unzip it into the data folder.
    
    wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    unzip wiki-pages.zip -d data
 

#### 2. Indexing 
Construct an SQLite Database (go grab a coffee while this runs)

    PYTHONPATH=src python src/scripts/build_db.py data/wiki-pages data/fever/fever.db

 
### Download the UKP-Athene models
* Download the datasets already processed through document retrieval, the pre-trained sentence selection ESIM model and the pre-trained claim verification ESIM models [here](https://drive.google.com/drive/folders/17ibwpWeozFUORODduMKMNDTF0XM1_GgH?usp=sharing). Then place the datasets and the models as following:


    mkdir -p model/no_attention_glove/rte_checkpoints/
    mkdir -p model/esim_0/rte_checkpoints/
    mkdir -p model/esim_0/sentence_retrieval_ensemble/
    unzip claim_verification_esim.ckpt.zip -d model/no_attention_glove/rte_checkpoints/
    unzip claim_verification_esim_glove_fasttext.ckpt.zip -d model/esim_0/rte_checkpoints/
    unzip sentence_retrieval_ensemble.ckpt.zip -d model/esim_0/sentence_retrieval_ensemble/
    unzip document_retrieval_datasets.zip -d data/fever/    
    
### Run the end-to-end pipeline of the submitted models

    PYTHONPATH=src python src/script/athene/pipeline.py
    
### Run the pipeline in different modes:
Launch the pipeline with optional mode arguments:

    PYTHONPATH=src python src/script/athene/pipeline.py [--mode <mode>]

All possible modes are as followings:

|Modes|Description|
|---|---|
| `PIPELINE` | Default option. Run the complete pipeline with both training and predicting phases. |
| `PIPELINE_NO_DOC_RETR` | Skip the document retrieval sub-task. With both training and predicting phases. In this case, the datasets already processed by document retrieval are needed.|
| `PIPELINE_RTE_ONLY` | Train and predict only for the RTE sub-task. In this case, the datasets already processed by sentence retrieval are needed.|
| `PREDICT` | Run the all 3 sub-tasks, but only with predicting phase in sentence retrieval and RTE. Only the test set is processed by the document retrieval and sentence retrieval sub-tasks.|
| `PREDICT_NO_DOC_RETR` | Skip the document retrieval sub-task, and only with predicting phase. The test set processed by the document retrieval is needed, and only the test set is processed by sentence retrieval sub-tasks.|
| `PREDICT_RTE_ONLY` | Predict for only the RTE sub-task.|
| `PREDICT_ALL_DATASETS` | Run the all 3 sub-tasks, but only with predicting phase in sentence retrieval and RTE. All 3  datasets are processed by the document retrieval and sentence retrieval sub-tasks.|
| `PREDICT_NO_DOC_RETR_ALL_DATASETS` | Skip the document retrieval sub-task, and only with predicting phase. All 3 datasets processed by the document retrieval are needed. All 3 datasets are processed by sentence retrieval sub-tasks.|
    

### Run the variation of the RTE model
Another variation of the ESIM model is configured through the config file in the conf folder.

To run the models:
    
    PYTHONPATH=src python src/scripts/athene/pipeline.py --config conf/<config_file> [--mode <mode>]
    
### Contacts:
If you have any questions regarding the code, please, don't hesitate to contact the authors or report an issue.
  * \<lastname\>@ukp.informatik.tu-darmstadt.de
  * https://www.informatik.tu-darmstadt.de/ukp/ukp_home/
  * https://www.tu-darmstadt.de    
  
### License:
  * Apache License Version 2.0