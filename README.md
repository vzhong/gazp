# README

This repo contains the source code for [Grounded Adaptation for Zero-shot Executable Semantic Parsing](https://arxiv.org/abs/2009.07396).
If you find this work helpful, please cite:

```bib
@inproceedings{ zhong2020grounded,
  title={ Grounded Adaptation for Zero-shot Executable Semantic Parsing },
  author={ Zhong, Victor and Lewis, Mike and Wang, Sida I. and Zettlemoyer, Luke },
  booktitle={ EMNLP },
  year={ 2020 }
}
```

First, download the data such that the `data` folder has:

```
cosql
database
sparc
spider
tables.json
```

Next, install dependencies:

```
pip install -r requirements.txt
mkdir -p cache
bash download.sh
```

Download NLTK:

```python
>>> import nltk
>>> nltk.download('punkt')
```

Next, preprocess data:

```
python preprocess_nl2sql.py
python preprocess_sql2nl.py
```

# Training NL to SQL

```bash
python train.py --dataset spider --name r1 --keep_values --model nl2sql
```

# Training SQL to NL

Wait until `nl2sql` is done training because we will use it to do early-stopping.

```bash
python train.py --dataset spider --name r1 --model sql2nl --fparser exp/nl2sql/r1/best.tar
```

# Generate data

```bash
mkdir gen
python generate.py --num 50000 --fout gen/gen1.json --resume exp/sql2nl/r1/best.tar --fparser exp/nl2sql/r1/best.tar
python generate.py --num 50000 --fout gen/gen2.json --resume exp/sql2nl/r1/best.tar --fparser exp/nl2sql/r1/best.tar --seed 2
```

# Retrain using generated data

```bash
python train.py --dataset spider --name r2 --keep_values --model nl2sql --aug gen/gen1.pt gen/gen2.pt
```

You should see improvements along the lines of

```bash
vzhong@uw ~/p/g/e/nl2sql > tail -n15  */train.best.json
==> r1/train.best.json <==
  "epoch": 37,
  "iteration": 282112,
  "loss_query": 0.014852346881472668,
  "loss_value": 0.009498934775701118,
  "time_forward": 0.21286849603357738,
  "time_backward": 0.3552022272292249,
  "time_batch": 0.5680707232628023,
  "train_em": 0.2390894396551724,
  "train_bleu": 0.5972011524320889,
  "train_official_em": 0.2390894396551724,
  "dev_em": 0.10810810810810811,
  "dev_bleu": 0.42340510532533265,
  "dev_official_em": 0.5424710424710425,
  "dev_official_ex": 0.528957528957529
}
==> r2/train.best.json <==
  "epoch": 28,
  "iteration": 1385765,
  "loss_query": 0.014741608709535123,
  "loss_value": 0.009798103380507111,
  "time_forward": 0.17836320367810315,
  "time_backward": 0.3074339982679534,
  "time_batch": 0.4857972019460566,
  "train_em": 0.5037145547766035,
  "train_bleu": 0.6272792293851236,
  "train_official_em": 0.5037145547766035,
  "dev_em": 0.11969111969111969,
  "dev_bleu": 0.43715406265853546,
  "dev_official_em": 0.5598455598455598,
  "dev_official_ex": 0.5752895752895753
```

For Sparc and CoSQL, the procedure is similar, except you should use the corresponding files:

```
preprocess_(nl2sql|sql2nl)_(sparc|cosql).py
```

When training, you should use the flag `--interactive_eval` to perform student forcing during evaluation.
For more options, see `python train.py --help`.
When training and generating (with `generate.py`), you should set the correct `--dataset`.


These numbers are slightly different from those in the paper due to different versions of PyTorch etc (this release should work with the latest at the time of release, which is 1.7).
Alternatively, I have also included checkpoints used in my submission.
You can [download these here](https://drive.google.com/file/d/1aA7z27UySNFTlWxuX6rWlBTKBtHps6RR/view?usp=sharing) (it is rather large, at 6.9 GB zipped).
You should unzip this in the root folder and then adapt (eg. synthesize data and retrain using these checkpoints instead of `r1`) using these pairs of models.
To sanity check the download, you should be able to run `python train.py --test_only --dataset spider --drnn 300 --keep_values --model nl2sql --resume best/spider_nl2sql.tar` and see an `official_em` of 54.83.
Remeber, should you want to apply GAZP to your model on the official evaluation, **you need to do adaptation on the test DBs!**


If you have questions, please make a Github issue.
For comments about the paper, please email me at [victor@victorzhong.com](mailto:victor@victorzhong.com).


## If you'd like to help out with this codebase

Admittedly this codebase is on the more difficult to navigate side.
Since the release of our paper, more capable parsers have been published.
Some of these pareser can also generate full SQL queries (e.g. [Salesforce Research's work here](https://github.com/salesforce/TabularSemanticParsing)).
I would love it if someone could apply the ideas in GAZP to a more powerful parser (ideally without the excruciating preprocessing steps that both GAZP and such parsers typically require).
If you have ideas on how to do this, please get in touch with me!
Otherwise, I hope I will have time to do this eventually.
