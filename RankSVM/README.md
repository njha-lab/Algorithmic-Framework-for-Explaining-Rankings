\# RankSVM Baseline



This directory contains the RankSVM baseline used for comparison with

our methods for finding linear explanations of a given ranking.



The underlying RankSVM implementation is based on SVMrank by

Thorsten Joachims. We provide additional Python scripts for:



\- converting CSV datasets into the SVMrank input format,

\- running RankSVM,

\- extracting the learned weight vector,

\- normalizing the learned weight vector,

\- supporting full-ranking and Top-k settings,

\- varying the number of tuples considered, and

\- evaluating the learned ranking using PP, INV, MPE, and LIPE.



\## Directory Structure



&#x20;   RankSVM/

&#x20;   |

&#x20;   |-- Scripts/

&#x20;   |   |-- config.py

&#x20;   |   |-- converter.py

&#x20;   |   |-- csv\_to\_svmrank.py

&#x20;   |   |-- evaluation.py

&#x20;   |   `-- run\_svmrank.py

&#x20;   |

&#x20;   |-- Datasets/

&#x20;   |   `-- example.csv

&#x20;   |

&#x20;   |-- Models/

&#x20;   |-- Predictions/

&#x20;   |-- Results/

&#x20;   |

&#x20;   |-- svm\_rank/

&#x20;   |   |-- svm\_rank\_learn.exe

&#x20;   |   `-- svm\_rank\_classify.exe

&#x20;   |

&#x20;   |-- svm\_light/

&#x20;   |-- svm\_struct/

&#x20;   |

&#x20;   |-- svm\_struct\_api.c

&#x20;   |-- svm\_struct\_api.h

&#x20;   |-- svm\_struct\_api\_types.h

&#x20;   |-- svm\_struct\_learn\_custom.c

&#x20;   |

&#x20;   |-- Makefile

&#x20;   |-- build\_linux.sh

&#x20;   |-- requirements.txt

&#x20;   `-- LICENSE.txt



\## Requirements



The Python pipeline requires Python 3 and the packages listed in

`requirements.txt`.



Install the dependencies using:



&#x20;   pip install -r requirements.txt



\## Dataset Format



Input datasets are provided as CSV files.



The user specifies:



\- the dataset,

\- feature columns,

\- ranking column,

\- whether the ranking is full or Top-k,

\- k for Top-k experiments, and

\- n when only a specified number of tuples should be considered.



The CSV file is automatically converted into the format required by

SVMrank.



\## Running Example



`Datasets/example.csv` contains the running example with three features:



&#x20;   SAT

&#x20;   GRE

&#x20;   TOEFL



The ground-truth ranking is given by the column:



&#x20;   Rank



For this dataset, rank 1 represents the highest-ranked tuple.



\### Full Ranking



On Windows:



&#x20;   python Scripts\\run\_svmrank.py --dataset example.csv --features SAT GRE TOEFL --ranking Rank --mode full



\### Varying the Number of Tuples



Use the `--n` parameter to restrict the experiment to the first n

tuples of the ranking.



For example:



&#x20;   python Scripts\\run\_svmrank.py --dataset example.csv --features SAT GRE TOEFL --ranking Rank --mode full --n 5



\### Top-k Ranking



For a Top-k experiment, use `--mode topk` and specify k:



&#x20;   python Scripts\\run\_svmrank.py --dataset example.csv --features SAT GRE TOEFL --ranking Rank --mode topk --k 5



\## Output



RankSVM learns a linear scoring function of the form



&#x20;   f\_W(x) = W^T x.



The learned weight vector is extracted from the SVMrank model and

normalized before evaluation.



The implementation evaluates the learned weight vector using the

following four error measures:



\- PP: Pairwise Penalty

\- INV: Inversions

\- MPE: Minimum Position Error

\- LIPE: Largest Individual Position Error



Generated files are stored in:



&#x20;   Models/

&#x20;   Predictions/

&#x20;   Results/



\## Linux / Ubuntu



The RankSVM C source code is included in this directory.



The repository also contains:



&#x20;   Makefile

&#x20;   build\_linux.sh



which can be used to compile the RankSVM executables on Linux.



After compilation, the Linux executables can be used instead of the

Windows `.exe` binaries.



\## SVMrank



The underlying SVMrank implementation was developed by Thorsten

Joachims.



Official SVMrank website:



https://www.cs.cornell.edu/people/tj/svm\_light/svm\_rank.html



Please refer to the original SVMrank distribution and `LICENSE.txt`

for the applicable licensing terms.

