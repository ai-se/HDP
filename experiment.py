#__author__ = 'WeiFu'
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
  for group, srclst in datasrc.iteritems():
    for one in srclst:
      random.seed(1)
      # one = "./dataset/MORPH/ant-1.3.arff"
      arffheader, arffcontent = readarff(one)
      out_wpdp, out_cpdp, out_hdp = [], [], []  # store results for three methods
      for _ in xrange(500):
        random.shuffle(arffcontent)
        cut = int(len(arffcontent) * 0.5)
        A = arffcontent[:cut]
        B = arffcontent[cut:]
        writearff(arffheader + "".join(A), "train")
        writearff(arffheader + "".join(B), "test")
        # out_wpdp += wpdp()
        #cpdp(group,one)
        # pdb.set_trace()
        out_hdp +=hdp(one,source_target_match)
      # pdb.set_trace()
      re_sorted = sorted(out_hdp)
      print(one, "===>", re_sorted[int(len(re_sorted) * 0.5)])
if __name__ == "__main__":
  run()