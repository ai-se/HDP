# __author__ = 'WeiFu'
from __future__ import print_function, division
import sys
import pdb
import random
from os import listdir
from os.path import isfile, join
from scipy import stats
import numpy as np


class o:
  def __init__(i, **d): i.__dict__.update(d)

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




def KSanalyzer():
  data = read()
  pdb.set_trace()
  for key, val in data.iteritems():
    for key1, val1 in data.iteritems():
      if key != key:
        for one in val:
          for one1 in val1:








if __name__ == "__main__":
  random.seed(1)
  np.random.seed(1)
  KSanalyzer()



