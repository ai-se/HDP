# __author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random, math
from utility import *
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
import numpy as np
import networkx as nx


def transform(d,selected = []):
  """
  the data will be stored by column, not by instance
  :param d : data
  :type d: o
  :return col: the data grouped by each column
  :type col: dict
  """
  col = {}
  for row in d["data"]:
    for attr, cell in zip(d["attr"][:-1], row[:-1]):  # exclude last columm, $bug
      if len(selected) != 0 and attr not in selected:  # get rid of name, version columns.
        continue # if this is for feature selected data, just choose those features.
      col[attr] = col.get(attr, []) + [cell]
  return col


def maximumWeighted(match, target_lst, source_lst):
  """
  using max_weighted_bipartite to select a group of matched metrics
  :param match : matched metrics with p values, key is the tuple of matched metrics
  :type match : dict
  :param target_lst : matched target metrics
  :type target_lst: list
  :param source_lst : matched source metcis
  :type source_lst: list
  :return : matched metrics as well as corresponding values
  :rtype: class o
  """

  value = 0
  attr_source, attr_target = [], []
  G = nx.Graph()
  for key, val in match.iteritems():
    G.add_edge(key[0] + "source", key[1] + "target", weight=val)  # add suffix to make it unique
  Result = nx.max_weight_matching(G)
  for key, val in Result.iteritems():  # in Results, (A:B) and (B:A) both exist
    if key[:-6] in source_lst and val[:-6] in target_lst:
      attr_target.append(val[:-6])
      attr_source.append(key[:-6])
      value += match[(key[:-6], val[:-6])]
  # pdb.set_trace()
  return o(score=value, attr_source=attr_source, attr_target=attr_target)


def KStest(d1, d2, sourcename, cutoff=0.05):
  """
  Kolmogorov-Smirnov Test
  :param d1 : source data
  :type d1 : o
  :param d2: target data
  :type d2: o
  :param sourcename: src of source
  :type sourcename : str
  :return : results of maximumWeighted
  :rtype: o
  """
  match = {}
  # pdb.set_trace()
  A = loadWekaData(sourcename)
  A_selected = featureSelection(A, int((A.class_index) * 0.15))
  features = [str(i).split(" ")[1] for i in A_selected.attributes()][:-1]
  source = transform(d1,features)
  target = transform(d2)
  # pdb.set_trace()
  target_lst, source_lst = [], []
  for tar, val1 in target.iteritems():
    for sou, val2 in source.iteritems():
      result = stats.ks_2samp(np.array(val1), np.array(val2))  # (a,b): b is p-value, zero means significantly different
      if result[1] > cutoff:
        # match[sou] = match.get(sou,[])+[(tar,result[1])]
        match[(sou, tar)] = result[1]
        if tar not in target_lst:
          target_lst.append(tar)
        if sou not in source_lst:
          source_lst.append(sou)
  return maximumWeighted(match, target_lst, source_lst)


def KSanalyzer(data=read()):
  """
  for each target data set, find a best source data set in terms of p-values
  :param data : read csv format of data
  :type data : o
  :return pairs of matched data
  :rtype: list
  """
  # data = read()
  best_pairs = []
  for key, val in data.iteritems():
    for target in val:
      temp_score = 0
      temp_best = None
      for key1, val1 in data.iteritems():
        if key != key1:
          for source in val1:
            source_name = "./dataset/" + key1 + "/" + source["name"][
                                                      source["name"].rfind("/") + 1:source["name"].rfind(".")] + ".arff"
            target_name = target["name"][target["name"].rfind("/") + 1:target["name"].rfind(".")]
            X = KStest(source, target,source_name).update(train_src=source_name, test_src=target_name)
            # if X["score"] > temp_score:
            #   temp_score = X["score"]
            #   temp_best = X # it seems they use all feasible pairs, not the best one as I thought
            best_pairs.append(X)
      # pdb.set_trace()
  return best_pairs


def call(train_src, test_src, train_attr, test_attr):
  """
  call weka to perform learning and testing
  :param train: src of training data
  :type train: str
  :param test: src of testing data
  :type test: str
  :param train_attr: matched feature for training data set
  :type train_attr: list
  :param test_attr: matched feature for testing data set
  :type test_attr: list
  :return ROC area value
  :rtype: list
  """
  r = round(wekaCALL(train_src, test_src, train_attr, test_attr, True), 3)
  if not math.isnan(r):
    return [r]
  else:
    return []


def hdp(test_src, source_target_match):
  """
   source_target_match = KSanalyzer()
  :param test_src : src of test(target) data set
  :type test_src : str
  :param source_target_match : matched source and target data test
  :type source_target_match: list
  :param test_A : first half-split test data in wpdp
  :type test_A : Instance
  :param test_B : second half-split test data in wpdp
  :type test_B : Instance
  :return: value of ROC area
  :rtype: list
  """
  result1, result2, result = [],[],[]
  test_name = test_src[test_src.rfind("/") + 1:test_src.rfind(".")]
  for i in source_target_match:
    if i.test_src == test_name:  # for all
      train_attr = i.attr_source
      test_attr = i.attr_target
      train_src = i.train_src
      result1 += call(train_src, "./exp/train.arff", train_attr, test_attr)  # hdp should use the same test data splits as wpdp
      result2 += call(train_src, "./exp/test.arff", train_attr, test_attr)  # test.arff and train.arff are both test case for hdp
  # if train_src == "": return []
  A = sorted(result1)
  B = sorted(result2)
  result +=[A[-1]]
  result +=[B[-1]]
  # print(result)
  return result


def testEQ():
  def tofloat(lst):
    for x in lst:
      try:
        yield float(x)
      except ValueError:
        yield x[:-1]

  test_src = "./dataset/Relink/apache.arff"
  train_src = "./dataset/AEEEM/EQ.arff"

  d = open("./datasetcsv/AEEEM/EQ.csv", "r")
  content = d.readlines()
  attr = content[0].split(",")
  inst = [list(tofloat(row.split(","))) for row in content[1:]]
  d1 = o(name="./datasetcsv/AEEEM/EQ.csv", attr=attr, data=inst)

  d = open("./datasetcsv/Relink/apache.csv", "r")
  content = d.readlines()
  attr = content[0].split(",")
  inst = [list(tofloat(row.split(","))) for row in content[1:]]
  d2 = o(name="./datasetcsv/Relink/apache.csv", attr=attr, data=inst)
  Result = KStest(d1,d2,train_src)
  print(Result)
  pdb.set_trace()
  print("DONE")


if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  # wpdp()
  # KSanalyzer()
  # wekaCALL()
  # filter()
  # cpdp()
  # readarff()
  testEQ()



