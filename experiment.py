# __author__ = 'WeiFu'
from __future__ import division, print_function
import sys
import pdb
import random
from utility import *
from wpdp import *
from cpdp import *
from hdp import *


def run():
  datasrc = readsrc()
  source_target_match = KSanalyzer()
  # pdb.set_trace()
  for group, srclst in datasrc.iteritems():
    for one in srclst:
      random.seed(1)
      data = loadWekaData(one)
      out_wpdp, out_cpdp, out_hdp = [], [], []  # store results for three methods
      for _ in xrange(500):
        randomized = filter(data, False,"","weka.filters.unsupervised.instance.Randomize", ["-S", str(_)])
        train = filter(randomized, True,"train","weka.filters.supervised.instance.StratifiedRemoveFolds",["-N", "2", "-F", "1", "-S", "1"])
        test = filter(randomized, True,"test","weka.filters.supervised.instance.StratifiedRemoveFolds",["-N", "2", "-F", "2", "-S", "1"])
        # pdb.set_trace()
        # out_wpdp += wpdp(tarin, test)
        #cpdp(group,one)
        temp = hdp(one, source_target_match)
        # print(sorted(temp))
        if len(temp) == 0:
          continue
        else:
          out_hdp += temp
      # pdb.set_trace()
      re_sorted = sorted(out_hdp)
      print(one, "===>", re_sorted[int(len(re_sorted) * 0.5)])
      # pdb.set_trace()
      # print("next=======>")


if __name__ == "__main__":
  run()