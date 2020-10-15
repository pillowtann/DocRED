import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython

# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'CNN3', help = 'name of the model')
parser.add_argument('--save_name', type = str, default = 'CNN3')

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--output_file', type = str, default = "result.json")


args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	# 'LSTM': models.LSTM,
	# 'BiLSTM': models.BiLSTM,
	# 'ContextAware': models.ContextAware,
	# 'LSTM_SP': models.LSTM_SP
}

con = config.Config(args)

# limited gpu ram
con.set_batch_size(10)
# run only 3 rounds
con.set_max_epoch(5)
# run in x number of epoch
# con.set_test_epoch(1)
# use the whole dataset
con.set_train_batches(None)
# load previous model
con.set_pretrain_model(os.path.join('./checkpoint', args.save_name))

con.load_train_data()
con.load_test_data()
# con.set_train_model()
con.train(model[args.model_name], args.save_name)
