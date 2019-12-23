## Early Rumour Detection
Rumours can spread quickly through social media, and malicious ones 
can bring about significant economical and social impact.  Motivated by 
this, our paper focuses on the task of rumour detection; particularly, 
we are interested in understanding how early we can detect 
them. To address this, we present a novel methodology for early rumour 
detection.Here is the code based on our approach.

### Requirement
Python 3.6

TensorFlow  1.13

### DataSet

Two DataSets can be used to evaluate our model.

Weibo DataSet: http://alt.qcri.org/~wgao/data/rumdect.zip

Twitter DataSet: https://figshare.com/articles/PHEME_dataset_of_rumours_and_non-rumours/4010619

### Usage
1. Download Twitter DataSet and extract, set the DataSet path to the `data_file_path` in `config.py`.

2. Download glove word vectors: http://nlp.stanford.edu/data/glove.840B.300d.zip, and set the `w2v_file_path` in `config.py`.

3. Run `python main.py` to train and evaluate the model.

## Early Rumour Detection (Torch)
If there are problems with the code, you can try the newly uploaded torch codes.
