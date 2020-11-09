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
parser.add_argument('--load_model', type = str, default = None)

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--output_file', type = str, default = "result.json")

parser.add_argument('--max_epoch', type = int, default = 20)
parser.add_argument('--batch_size', type = int, default = 40)

parser.add_argument('--tf_train_batches', type = bool, default = False)
parser.add_argument('--train_batches', type = int, default = 20)
parser.add_argument('--tf_limit', type = bool, default = False)
parser.add_argument('--limit', type = float, default = 0.5)

args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	'CNN3-coref': models.CNN3coref,
	'CNN3-ner': models.CNN3ner,
	'CNN3-distance': models.CNN3distance,
	'CNN3-all': models.CNN3all,
	# 'LSTM': models.LSTM,
	# 'BiLSTM': models.BiLSTM,
	# 'ContextAware': models.ContextAware,
	# 'LSTM_SP': models.LSTM_SP
}

con = config.Config(args)

# limited gpu ram
con.set_batch_size(args.batch_size)
# run only 3 rounds
con.set_max_epoch(args.max_epoch)
# run in x number of epoch
# con.set_test_epoch(1)
# use the whole dataset
if args.tf_train_batches:
	con.set_train_batches(args.train_batches)
else:
	con.set_train_batches(None)
if args.tf_limit:
	con.set_limit_train_batches(args.limit)
else:
	con.set_limit_train_batches(None)
# load previous model
if args.load_model is not None:
	con.set_pretrain_model(args.load_model)

con.load_train_data()
con.load_test_data()
# con.set_train_model()
con.train(model[args.model_name], args.save_name)
