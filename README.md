# deep-distant-supervision
Distant supervision algorithms with neural network.

# Model

## Training
To train the deep-distant supervision model, try

`python main.py --is_train=True --num_hidden=256 --data_set=data/nyt --num_epoch=3 --save_gap=1000 --print_gap=100`

Examples of some configurable model parameters are:
- `word_attn`: boolean, whether to use word-level attention layer
- `sent_attn`: boolean, whether to use sentence-level attention layer
- `bidirectional`: boolean, whether to use bidirectional rnn 
- `data_set`: path to dataset folder. the folder should contain `train_x.npy` and `train_y.npy` for training
- `save_gap`: the number of batch steps used to save partially trained models.
- `print_gap`: print status of training for every `print_gap`

To see the all possible configurations, try

`python main.py --help`

### Data format
`train_x.npy` and `train_y.npy` are preprocessed training data. These files should be in the data folder set by `data_set` parameter.

`train_x.npy` contains a `list` of triples, where each triple consists of a `list` of sentences.

A sentence is a list of tuples with `(token_id, pos_en1, pos_en2)`
- `token_id`: ID of word token in a dictionary
- `pos_en1`': relative position of the token to entity 1
- `pos_en2`': relative position of the token to entity 2

`train_y.npy` contains a list of one-hot vectors for triples. The order of triples is the same as `train_x.npy`.

## Testing
To test the trained model, try

`python main.py --is_train=False --data_set=data/nyt`

# Requirements
- Tensorflow 1.2
- numpy, tqdm, scikit-learn

# References
- https://github.com/thunlp/TensorFlow-NRE
