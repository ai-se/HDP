# __author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random,math
from utility import *
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
import numpy as np
import hungarian
import networkx as nx
import matplotlib.pyplot as plt

def transform(d):
  col = {}
  for row in d["data"]:
    for attr, cell in zip(d["attr"][:-1], row[:-1]):  # exclude last columm, $bug
      if "?" not in attr: # get rid of name, version columns.
        col[attr] = col.get(attr, []) + [cell]
  return col


def maximumWeighted(match, target_lst, source_lst):
  value = 0
  attr_source, attr_target = [],[]
  G = nx.Graph()
  for key, val in match.iteritems():
    G.add_edge(key[0],key[1], weight= val)
  Result = nx.max_weight_matching(G)
  for key,val in Result.iteritems(): # in Results, (A:B) and (B:A) both exist
    if val in target_lst:
      attr_target.append(val)
      attr_source.append(key)
      value += match[(key,val)]
  # pdb.set_trace()
  return  o(score=value, attr_source=attr_source, attr_target=attr_target)
# def maximumWeighted(match, target_lst, source_lst):
#   value, count = 0, 0
#   attr_source, attr_target,grid = [],[],[]
#   for key in source_lst:
#     row = []
#     for tar in target_lst:
#       found = False
#       for one in match[key]:
#         if tar in one:
#           row +=[one[1]]
#           found = True
#       if not found:
#         row += [-100]
#     grid.append(row)
#   print(grid)
#   total, coordinates = hungarian.maximize(grid)
#   for (row, col) in coordinates:
#     attr_source.append(source_lst[row])
#     attr_target.append(target_lst[col])
#     print(source_lst[row],"==>",target_lst[col])
#
#   pdb.set_trace()
#
#   # for a in metrics:
#   #   if count < top:  # select top 15% features
#   #     if a[0][1] not in attr_source and a[0][0] not in attr_target:
#   #       value += a[-1]
#   #       attr_source.append(a[0][1])
#   #       attr_target.append(a[0][0])
#   #       count += 1
#   return o(score=value, attr_source=attr_source, attr_target=attr_target)


def KStest(d1, d2, cutoff=0.05):
  match = {}
  source = transform(d1)
  target = transform(d2)
  target_lst, source_lst = [], []
  for tar, val1 in target.iteritems():
    for sou, val2 in source.iteritems():
      result = stats.ks_2samp(np.array(val1), np.array(val2))  # (a,b): b is p-value, zero means significantly different
      if result[1] >= cutoff:
        # match[sou] = match.get(sou,[])+[(tar,result[1])]
        match[(sou,tar)] = result[1]
        if tar not in target_lst:
          target_lst.append(tar)
        if sou not in source_lst:
          source_lst.append(sou)
  return maximumWeighted(match, target_lst, source_lst)


def KSanalyzer(data = read()):
  """
  for each target data set, find a best source data set in terms of p-values
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
            source_name =  "./dataset/"+key1+"/"+source["name"][source["name"].rfind("/")+1:source["name"].rfind(".")]+".arff"
            target_name =  target["name"][target["name"].rfind("/")+1:target["name"].rfind(".")]
            X = KStest(source, target).update(train_src=source_name, test_src=target_name)
            if X["score"] > temp_score:
              temp_score = X["score"]
              temp_best = X
      best_pairs.append(temp_best)
  print(best_pairs)
  # pdb.set_trace()
  return best_pairs

def call(train,test,train_attr,test_attr):
  r = round(wekaCALL(train,test,train_attr,test_attr,True),3)
  if not math.isnan(r):
    return r
  else:
    return 0

def hdp(test_src, source_target_match):
  # source_target_match = KSanalyzer()
  result = []
  train_src = ""
  test_name = test_src[test_src.rfind("/")+1:test_src.rfind(".")]
  train_attr,test_attr = [],[]
  for i in source_target_match:
    if i.test_src == test_name:
      train_src = i.train_src
      train_attr = i.attr_source
      test_attr = i.attr_target
  # pdb.set_trace()
  # train_src ='./dataset/AEEEM/EQ.arff'
  # train_attr =['ck_oo_noc', 'WCHU_noc', 'numberOfMajorBugsFoundUntil:', 'ck_oo_numberOfAttributes', 'ck_oo_numberOfPrivateAttributes', 'WCHU_rfc', 'ck_oo_numberOfPublicMethods', 'ck_oo_numberOfPrivateMethods', 'ck_oo_fanOut']
  # test_attr = ['blank_loc', 'code_and_comment_loc', 'formal_parameters', 'decision_count', 'design_complexity', 'call_pairs', 'cyclomatic_complexity', 'multiple_condition_count', 'branch_count']
  result +=[call(train_src,"./exp/test.arff", train_attr, test_attr)] # hdp should use the same test data splits as wpdp
  result +=[call(train_src,"./exp/train.arff",train_attr,test_attr)] # test.arff and train.arff are both test case for hdp
  return result





def prepareData(train,test):
  train_x = [t[:-1] for t in train]
  train_y = [(t[-1]) for t in train]
  test_x = [t[:-1] for t in test]
  test_y = [(t[-1]) for t in test]
  return [train_x, train_y, test_x, test_y]


def learner(d):
  train_x,train_y, test_x,test_y = d[0],d[1], d[2],d[3]
  train_y = [1  if i == "buggy"  else 0 for i in d[1]]
  test_y = [1  if i == "buggy"  else 0 for i in d[3]]
  lr = LogisticRegression()
  try:
    clf = lr.fit(train_x,train_y)
  except ValueError:
    return 0
  # y_predict = clf.predict(test_x)
  y_score = clf.decision_function(test_x)
  fpr, tpr, _ = roc_curve(test_y,y_score) #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
  roc_auc1 = auc(fpr, tpr)
  # roc_auc = roc_auc_score(test_y, y_score) #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
  return roc_auc1






if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  # wpdp()
  # KSanalyzer()
  #wekaCALL()
  # filter()
  # cpdp()
  readarff()



