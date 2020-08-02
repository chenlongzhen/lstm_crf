# -*-encoding=utf8-*-
from flask import jsonify
from flask import Flask
from flask import request
import platform
import codecs
import logging
import itertools
from collections import OrderedDict
import os
import sys
from gevent import monkey

monkey.patch_all(thread=False)
from gevent import pywsgi
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, load_config, create_model, save_config, save_model
from utils import make_path

from data_utils import load_word2vec, create_input, input_from_line, BatchManager

currentPath = os.getcwd()
sys.path.append(currentPath)

root_path = os.getcwd()
global pyversion
if sys.version > '3':
    pyversion = 'three'
else:
    pyversion = 'two'
if pyversion == 'three':
    import pickle
else:
    import cPickle, pickle
root_path = os.getcwd() + os.sep
print(root_path)
flags = tf.app.flags
flags.DEFINE_boolean("clean", False, "clean train folder")
flags.DEFINE_boolean("train", False, "Whether train the model")
# configurations for the model
flags.DEFINE_integer("seg_dim", 20, "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters")
flags.DEFINE_integer("lstm_dim", 100, "Num of hidden units in LSTM, or num of filters in IDCNN")
flags.DEFINE_string("tag_schema", "iobes", "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip", 5, "Gradient clip")
flags.DEFINE_float("dropout", 0.5, "Dropout rate")
flags.DEFINE_float("batch_size", 20, "batch size")
flags.DEFINE_float("lr", 0.001, "Initial learning rate")
flags.DEFINE_string("optimizer", "adam", "Optimizer for training")
flags.DEFINE_boolean("pre_emb", True, "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros", True, "Wither replace digits with zero")
flags.DEFINE_boolean("lower", False, "Wither lower case")

flags.DEFINE_integer("max_epoch", 100, "maximum training epochs")
flags.DEFINE_integer("steps_check", 100, "steps per checkpoint")
flags.DEFINE_string("ckpt_path", "ckpt2", "Path to save model")
flags.DEFINE_string("summary_path", "summary", "Path to store summaries")
flags.DEFINE_string("log_file", "train.log", "File for log")
flags.DEFINE_string("map_file", "maps.pkl", "file for maps")
flags.DEFINE_string("vocab_file", "vocab.json", "File for vocab")
flags.DEFINE_string("config_file", "config_file", "File for config")
flags.DEFINE_string("script", "conlleval", "evaluation script")
flags.DEFINE_string("result_path", "result", "Path for results")
flags.DEFINE_string("emb_file", os.path.join(root_path + "data", "vec.txt"), "Path for pre_trained embedding")
flags.DEFINE_string("train_file", os.path.join(root_path + "data", "example.train"), "Path for train data")
flags.DEFINE_string("dev_file", os.path.join(root_path + "data", "example.dev"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join(root_path + "data", "example.test"), "Path for test data")

# flags.DEFINE_string("model_type", "idcnn", "Model type, can be idcnn or bilstm")
flags.DEFINE_string("model_type", "bilstm", "Model type, can be idcnn or bilstm")

FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS.char_dim
    config["num_tags"] = len(tag_to_id)
    config["seg_dim"] = FLAGS.seg_dim
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size

    config["emb_file"] = FLAGS.emb_file
    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["pre_emb"] = FLAGS.pre_emb
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config


with open(FLAGS.map_file, "rb") as f:
    if pyversion == 'three':
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
    else:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f, protocol=2)
        # make path for store log and model if not exist
make_path(FLAGS)
if os.path.isfile(FLAGS.config_file):
    config = load_config(FLAGS.config_file)
else:
    config = config_model(char_to_id, tag_to_id)
    save_config(config, FLAGS.config_file)
make_path(FLAGS)
app = Flask(__name__)
log_path = os.path.join("log", FLAGS.log_file)
logger = get_logger(log_path)
tf_config = tf.ConfigProto()
sess = tf.Session(config=tf_config)
# sess.run(tf.global_variables_initializer())
model = create_model(sess,
                     Model,
                     FLAGS.ckpt_path,
                     load_word2vec, config, id_to_char, logger)


@app.route('/ner', methods=['POST', 'GET'])
def get_text_input():
    # http://127.0.0.1:5002/ner?inputStr="最开心"
    text = request.args.get('inputStr')
    # print(text)
    if len(text.strip()) > 0:
        aa = model.evaluate_line(sess, input_from_line(text, char_to_id), id_to_tag)
        return jsonify(aa)


if __name__ == "__main__":
    app.config['JSON_AS_ASCII'] = False
    app.run(host='127.0.0.1', port=5002)