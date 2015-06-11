# __author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random
from os import listdir
from os.path import isfile, join
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, roc_auc_score
from scipy import stats
import numpy as np


class o:
  ID = 0
  def __init__(i, **d):
    o.ID = i.id = o.ID+1
    i.update(**d)

  def update(i, **d): i.__dict__.update(d); return i

  def __getitem__(i, k): return i.__dict__[k]

  def __hash__(i): return i.id

  def __repr__(i):
    keys = [k for k in sorted(i.__dict__.keys()) if k[0] is not "_"]
    show = [":%s %s" % (k, i.__dict__[k]) for k in keys]
    return '{' + ' '.join(show) + '}'


def read(src="./datasetcsv"):
  """
  read data from csv files, return all data in a dictionary

  {'AEEEM':[{name ='./datasetcsv/SOFTLAB/ar6.csv'
             attributes=['ck_oo_numberOfPrivateMethods', 'LDHH_lcom', 'LDHH_fanIn'...]
             instances=[[.....],[.....]]},]
   'MORPH':....
   'NASA':....
   'Relink':....
   'SOFTLAB':....]
  }

  """

  def tofloat(lst):
    for x in lst:
      try:
        yield float(x)
      except ValueError:
        yield x[:-1]

  data = {}
  folders = [i for i in listdir(src) if not isfile(i) and i != ".DS_Store"]
  for f in folders:
    path = join(src, f)
    for val in [join(path, i) for i in listdir(path) if i != ".DS_Store"]:
      d = open(val, "r")
      content = d.readlines()
      attr = content[0].split(",")
      inst = [list(tofloat(row.split(","))) for row in content[1:]]
      data[f] = data.get(f, []) + [o(name=val, attr=attr, data=inst)]
  return data


def transform(d):
  col = {}
  for row in d["data"]:
    for attr, cell in zip(d["attr"][:-1], row[:-1]):  # exclude last columm, $bug
      if "?" not in attr: # get rid of name, version columns.
        col[attr] = col.get(attr, []) + [cell]
  return col


def maximumWeighted(match, top):
  metrics = sorted(match.items(), key=operator.itemgetter(1), reverse=True)
  value, count = 0, 0
  attr_source, attr_target = [], []
  for a in metrics:
    if count < top:  # select top 15% features
      if a[0][0] not in attr_source and a[0][1] not in attr_target:
        value += a[-1]
        attr_source.append(a[0][0])
        attr_target.append(a[0][1])
        count += 1
  return o(score=value, attr_source=attr_source, attr_target=attr_target)


def KStest(d1, d2, cutoff=0.05):
  match = {}
  source = transform(d1)
  target = transform(d2)
  for key1, val1 in source.iteritems():
    for key2, val2 in target.iteritems():
      result = stats.ks_2samp(np.array(val1), np.array(val2))  # (a,b): b is p-value, zero means significantly different
      if result[1] >= cutoff:
        match[(key1, key2)] = result[1]
  return maximumWeighted(match, int(len(source) * 0.15))


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
            X = KStest(source, target).update(name_source=source["name"], name_target=target["name"])
            if X["score"] > temp_score:
              temp_score = X["score"]
              temp_best = X
      best_pairs.append(temp_best)
  return best_pairs

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
  clf = lr.fit(train_x,train_y)
  # y_predict = clf.predict(test_x)
  y_score = clf.decision_function(test_x)
  fpr, tpr, _ = roc_curve(test_y,y_score) #http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
  roc_auc1 = auc(fpr, tpr)
  # roc_auc = roc_auc_score(test_y, y_score) #http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
  return roc_auc1

def wpdp(data = read()):
  roc_results = {}
  for key, val in data.iteritems():
    for one in val:
      re_temp = []
      for _ in xrange(500):
        instances = one["data"]
        random.shuffle(instances)
        cut = int(len(instances)*0.5)
        A = instances[:cut]
        B = instances[cut:]
        re_temp += [learner(prepareData(A,B))]
        re_temp += [learner(prepareData(B,A))]
      re_sort = sorted(re_temp)
      roc_results[one["name"]] = o(rawresult = re_sort,median = re_sort[int(len(re_sort)*0.5)] )
      pdb.set_trace()



if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  wpdp()
  # KSanalyzer()



