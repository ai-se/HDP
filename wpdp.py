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
  roc_results = {}
  datasrc = readsrc()
  for group,srclst in datasrc.iteritems():
    for one in srclst:
      random.seed(1)
      one = "./dataset/MORPH/ant-1.3.arff"
      arffheader, arffcontent = readarff(one)
      result_once = []
      for _ in xrange(500):
        random.shuffle(arffcontent)
        cut = int(len(arffcontent)*0.5)
        A = arffcontent[:cut]
        B = arffcontent[cut:]
        writearff(arffheader+"".join(A),"train")
        writearff(arffheader+"".join(B),"test")
        result_once+=[call("./exp/train.arff","./exp/test.arff")]
        result_once+=[call("./exp/test.arff","./exp/train.arff")]
      re_sorted = sorted(result_once)
      print(re_sorted[int(len(re_sorted)*0.5)])
      pdb.set_trace()
      print("Done")





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
      print(one["name"],"--->" ,roc_results[one["name"]].median)
      # pdb.set_trace()


if __name__ == "__main__":
  wpdp()
  # wekaExp()