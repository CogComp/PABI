# PABI
This is the code repository for the EMNLP paper [Foreseeing the Benefits of Incidental Supervision](https://cogcomp.seas.upenn.edu/page/publication_view/959).
If you use this code for your work, please cite
```
@inproceedings{HZNR21,
    author = {Hangfeng He and Mingyuan Zhang and Qiang Ning and Dan Roth},
    title = {{Foreseeing the Benefits of Incidental Supervision}},
    booktitle = {Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2021},
    url = "https://cogcomp.seas.upenn.edu/papers/HZNR21.pdf",
    funding = {LwLL, MURI, CwC},
}
```


## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python>=3.6\
pip install -r requirements.txt

## Code Organization

The code is organized as follows:
- bpp.py (CWBPP algorithm for learning with various inductive signals)
- run_ner.py (BERT for NER)
- run_squad.py (BERT for QA)


## Reproducing experiments
To reproduce the experiments for learning with various inductive signals:
```
sh run_experiments.sh
```

To reproduce the experiments for cross-domain signals:
```
sh run_xdomain_ner_experiments.sh
sh run_xdomain_qa_experiments.sh
```

