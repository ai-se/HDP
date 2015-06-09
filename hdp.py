# __author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random
from os import listdir
from os.path import isfile, join
import operator

from scipy import stats
import numpy as np


class o:
  def __init__(i, **d): i.update(**d)

  def update(i, **d): i.__dict__.update(d); return i

  def __getitem__(i, k): return i.__dict__[k]

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


def KSanalyzer():
  data = read()
  for key, val in data.iteritems():
    for target in val:
      for key1, val1 in data.iteritems():
        if key != key1:
          for source in val1:
            X = KStest(source, target).update(name_source=source["name"], name_target=target["name"])
            pdb.set_trace()
            print(X)


if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  KSanalyzer()



