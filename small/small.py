from __future__ import division
import sys

import random
r = random.random
any = random.choice

def ps(dims,bins,p=0.1,skew=2):
  space = bins**dims
  f1  = [r()**skew for _ in xrange(space)]
  all = sum(f1)
  f2  = [x/all for x in f1]
  return [p/space*x for x in f2]

def ns(dims=3,bins=2,p=0.1,skew=2):
  w = sum(ps(dims,bins,p,skew))
  n = 10
  found = 0
  while True:
    if found > 0.66: break
    if n > 1000: break
    n += 10
    found = 1 - ((1 - w) ** n)
  print " _%s, _%s, _%s,  %s" % (int(p*10),dims,bins,n)

print "p,dims,bins,n66"
for i in xrange(10000):
  if i % 50 == 0: sys.stderr.write('%s ' % i)
  ns(dims= any([2,3,4,5,6,7]),
     bins= any([2,3,4,5,6,7]),
     skew = any([1,1.1,1.25,1.5,2]),
     p   = any([0.1,0.2,0.3,0.4]))
