![2010-07-07_ukp_banner](https://user-images.githubusercontent.com/29311022/27184688-27629126-51e3-11e7-9a23-276628da2430.png)

![aiphes_logo - small](https://user-images.githubusercontent.com/29311022/27278631-2e19f99e-54e2-11e7-919c-f89ae0c90648.png)
![tud_weblogo](https://user-images.githubusercontent.com/29311022/27184769-65c6583a-51e3-11e7-90e0-12a4bdf292e2.png)


## This repository was constructed by team Athene for the FEVER shared task 1 (http://fever.ai/2018/task.html)
### The system reached the third rank in the overall results and first rank on the evidence recall sub-task

This repository builts upon the baseline system repository developed by the FEVER shared task organizers: (https://github.com/sheffieldnlp/fever-naacl-2018)

For more information see our paper: [UKP-Athene: Multi-Sentence Textual Entailment for Claim Verification](https://arxiv.org/pdf/1809.01479.pdf)
* BibTeX:
	
            @article{hanselowski2018ukp,
                      title={UKP-Athene: Multi-Sentence Textual Entailment for Claim Verification},
                      author={Hanselowski, Andreas and Zhang, Hao and Li, Zile and Sorokin, Daniil and Schiller, Benjamin and Schulz, Claudia and Gurevych, Iryna},
                      journal={arXiv preprint arXiv:1809.01479},
                      year={2018}
                    }






## Download Data
Download the FEVER dataset from [the website of the FEVER share task](https://sheffieldnlp.github.io/fever/data.html) into the data directory

    mkdir data
    mkdir data/fever-data
    
    #To replicate the paper, download paper_dev and paper_test files. These are concatenated for the shared task
    wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
    wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
    wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl
    
Download pretrained GloVe Vectors

    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip glove.6B.zip -d data/glove
    gzip data/glove/*.txt
    
Download pretrained Wiki FastText Vectors

    wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
    mkdir -p data/fasttext
    unzip wiki.en.zip -d data/fasttext

## Create Python Environment
    conda create -n fever python=3.6
    source activate fever
    pip install -r requirements.txt


Download NLTK Punkt Tokenizer

    python -c "import nltk; nltk.download('punkt')"

## Data Preparation
The data preparation consists of three steps: downloading the articles from Wikipedia, indexing these for the Evidence Retrieval and performing the negative sampling for training . 

### 1. Download Wikipedia data:

Download the pre-processed Wikipedia articles and unzip it into the data folder.
    
    wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    unzip wiki-pages.zip -d data
 

### 2. Indexing 
Construct an SQLite Database (go grab a coffee while this runs)

    PYTHONPATH=src python src/scripts/build_db.py data/wiki-pages data/fever/fever.db

 
## Run the end to end pipeline of the submitted models

    PYTHONPATH=src python src/script/athene/pipeline.py

## Run the variation of the RTE model
Another variation of the ESIM model is configured through the config file in the conf folder.

To run the models:
    
    PYTHONPATH=src python src/scripts/athene/pipeline.py --config conf/<config_file>