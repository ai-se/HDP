#__author__ = 'WeiFu'
from __future__ import print_function, division
import pdb
import random,math
from utility import *


def call(train,test):
  r = round(wekaCALL(train,test),3)
  if not math.isnan(r):
    return r
  else:
    return 0

def wpdp():
  result_once = []
  result_once+=[call("./exp/train.arff","./exp/test.arff")]
  result_once+=[call("./exp/test.arff","./exp/train.arff")]
  return result_once


  #
  #
  # roc_results = {}
  # datasrc = readsrc()
  # for group,srclst in datasrc.iteritems():
  #   for one in srclst:
  #     random.seed(1)
  #     # one = "./dataset/MORPH/ant-1.3.arff"
  #     arffheader, arffcontent = readarff(one)
  #
  #     for _ in xrange(500):
  #       random.shuffle(arffcontent)
  #       cut = int(len(arffcontent)*0.5)
  #       A = arffcontent[:cut]
  #       B = arffcontent[cut:]
  #       writearff(arffheader+"".join(A),"train")
  #       writearff(arffheader+"".join(B),"test")
  #
  #     re_sorted = sorted(result_once)
  #     print(one,"===>",re_sorted[int(len(re_sorted)*0.5)])



if __name__ == "__main__":
  wpdp()
  # wekaExp()