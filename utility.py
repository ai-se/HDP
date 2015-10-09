# __author__ = 'WeiFu'
from __future__ import print_function, division
import jnius_config

jnius_config.add_options('-Xrs', '-Xmx4096m')
jnius_config.set_classpath('.', '/Users/WeiFu/Github/HDP_Jython/jar/weka.jar', '/Users/FuWei/Github/HDP/jar/weka.jar',
                           '/Users/WeiFu/Github/HDP_Jython/jar/commons-math3-3.5/commons-math3-3.5.jar',
                           '/Users/FuWei/Github/HDP/jar/commons-math3-3.5/commons-math3-3.5.jar')
import pdb
import random
import os
from os import listdir
from os.path import isfile, join
from jnius import autoclass


class o:
  ID = 0

  def __init__(i, **d):
    o.ID = i.id = o.ID + 1
    i.update(**d)

  def update(i, **d): i.__dict__.update(d); return i

  def __getitem__(i, k): return i.__dict__[k]

  def __hash__(i): return i.id

  def __repr__(i):
    keys = [k for k in sorted(i.__dict__.keys()) if k[0] is not "_"]
    show = [":%s %s" % (k, i.__dict__[k]) for k in keys]
    return '{' + ' '.join(show) + '}'


def printm(matrix):
  s = [[str(e) for e in row] for row in matrix]
  lens = [max(map(len, col)) for col in zip(*s)]
  fmt = ' | '.join('{{:{}}}'.format(x) for x in lens)
  for row in [fmt.format(*row) for row in s]:
    print(row)


def enumerateToList(enum):
  result = []
  while enum.hasMoreElements():
    result.append(enum.nextElement().toString())
  return result


def read(src="./dataset"):
  """
  read data from arff files, return all data in a dictionary

  {'AEEEM':[{name ='./datasetcsv/SOFTLAB/ar6.csv'
             attributes=['ck_oo_numberOfPrivateMethods', 'LDHH_lcom', 'LDHH_fanIn'...]
             instances=[[.....],[.....]]},]
   'MORPH':....
   'NASA':....
   'Relink':....
   'SOFTLAB':....]
  }
  """
  data = {}
  folders = [i for i in listdir(src) if not isfile(i) and i != ".DS_Store"]
  for f in folders:
    path = join(src, f)
    for val in [join(path, i) for i in listdir(path) if i != ".DS_Store"]:
      arff = loadWekaData(val)
      attributes = [str(i)[str(i).find("@attribute") + len("@attribute") + 1:str(i).find("numeric") - 1] for i in
                    enumerateToList(arff.enumerateAttributes())]  # exclude the label
      columns = [arff.attributeToDoubleArray(i) for i in range(int(arff.classIndex()))]  # exclude the class label
      data[f] = data.get(f, []) + [o(name=val, attr=attributes, data=columns)]
  return data


def readsrc(src="./dataset"):
  """
  read all data files in src folder into dictionary,
  where subfolder src are keys, corresponding file srcs are values
  :param src: the root folder src
  :type src: str
  :return: src of all datasets
  :rtype: dictionary
  """
  data = {}
  subfolder = [join(src, i) for i in listdir(src) if not isfile(join(src, i))]
  for one in subfolder:
    data[one] = [join(one, i) for i in listdir(one) if isfile(join(one, i)) and i != ".DS_Store"]
  return data


def loadWekaData(src):
  source = autoclass('weka.core.converters.ConverterUtils$DataSource')(src)
  data = source.getDataSet()
  data.setClassIndex(data.numAttributes() - 1)
  return data


def wekaCALL(source_src, target_src, source_attr=[], test_attr=[], isHDP=False):
  """
  weka wrapper to train and test based on the datasets
  :param source_src: src of traininng data
  :type source_src: str
  :param target_src: src of testing data
  :type target_src: str
  :param source_attr: features selected for building a learner
  :type source_attr:list
  :param test_attr: features selected in target data to predict labels
  :type test_attr: list
  :param isHDP: flag
  :type isHDP:bool
  :return: AUC
  :rtype: float
  """

  def getIndex(data, used_attr):
    # pdb.set_trace()
    del_attr = []
    used_attr = [i[1:-1] if i[0] == "'" or i[0] == '"' else i for i in used_attr]
    for k, attr in enumerate(enumerateToList(data.enumerateAttributes())):
      temp = str(attr)[str(attr).find("@attribute") + len("@attribute") + 1:str(attr).find("numeric") - 1]
      if temp[0] == "'":
        temp = temp[1:-1]
      if temp not in used_attr:
        del_attr += [k]
    return del_attr

  def delAttr(data, index):
    order = sorted(index, reverse=True)
    for i in order[:]:  # delete from big index, except for the class attribute
      data.deleteAttributeAt(i)
    return data

  #
  # if '-0.443dit...' in test_attr:
  # pdb.set_trace()
  source_data = loadWekaData(source_src)
  target_data = loadWekaData(target_src)
  # cls = Classifier(classname="weka.classifiers.functions.Logistic")
  cls = autoclass('weka.classifiers.functions.Logistic')()
  if isHDP:
    # pdb.set_trace()
    source_del_attr = getIndex(source_data, source_attr)
    target_del_attr = getIndex(target_data, test_attr)
    source_data = delAttr(source_data, source_del_attr)
    target_data = delAttr(target_data, target_del_attr)
  cls.buildClassifier(source_data)
  eval = autoclass('weka.classifiers.Evaluation')(source_data)
  eval.evaluateModel(cls, target_data)
  # target_data.num_attributes
  # print(eval.percent_correct)
  # print(eval.summary())
  # print(eval.class_details())
  # print(eval.area_under_roc(1))
  return eval.areaUnderROC(1)


def filter(data, toSave=False, file_name="test", filter_name="weka.filters.unsupervised.attribute.Remove",
           option=["-R", "first-3,last"]):
  # remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options = option)
  # option = ["-N","2","-F","2","-S","1"]
  remove = None
  filter = autoclass('weka.filters.AllFilter')
  if toSave:  # removeFolds
    remove = autoclass('weka.filters.unsupervised.instance.RemoveFolds')()
  else:
    remove = autoclass('weka.filters.unsupervised.instance.Randomize')()
  remove.setOptions(option)
  remove.setInputFormat(data)
  # remove.input(data)
  filtered = filter.useFilter(data, remove)
  if toSave:
    saver = autoclass('weka.core.converters.ArffSaver')()
    saver.setInstances(filtered)
    saver.setFile(autoclass("java.io.File")("./exp/" + file_name + ".arff"))
    saver.writeBatch()
    # saver.save_file(filtered, "./exp/" + file_name + ".arff")
  # print(filtered)
  return filtered


def save(data_instance, src):
  saver = autoclass('weka.core.converters.ArffSaver')()
  saver.setInstances(data_instance)
  saver.setFile(autoclass("java.io.File")(src))
  saver.writeBatch()


def featureSelection(data, num_of_attributes):
  """
  feature selection
  :param data: data to do feature selection
  :type data : Instance
  :param num_of_attributes : # of attributes to be selected
  :type num_of_attributes : int
  :return: data with selected feature
  :rtype: Instance
  """
  search = autoclass('weka.attributeSelection.Ranker')()
  evaluator = autoclass('weka.attributeSelection.ReliefFAttributeEval')()
  attsel = autoclass('weka.attributeSelection.AttributeSelection')()
  search.setOptions(['-N', str(num_of_attributes)])
  attsel.setSearch(search)
  attsel.setEvaluator(evaluator)
  attsel.SelectAttributes(data)
  return attsel.selectedAttributes()[:num_of_attributes] # this is the place where has a bug.


def selectInstances(old_data, option):
  """
  :param old_data: the data to be selected
  :type old_data: o
  :para option: parameters for filter
  :type option: list
  """
  arff = loadWekaData(old_data.name)  # re-read the data
  numInstance = arff.numInstances()
  while numInstance > option[option.index("-N") + 1]:
    random_index = random.randint(0, numInstance - 1)
    arff.remove(random_index)
    numInstance -= 1
  attributes = [str(i)[str(i).find("@attribute") + len("@attribute") + 1:str(i).find("numeric") - 1] for i in
                enumerateToList(arff.enumerateAttributes())]  # exclude the label
  columns = [arff.attributeToDoubleArray(i) for i in range(int(arff.classIndex()))]  # exclude the class label
  return o(name=old_data.name, attr=attributes, data=columns)


def PCA(data_src="", number_of_componets=2):
  def createfolder(src):
    new_src = "./R" + src
    if os.path.exists(new_src):
      return
    else:
      os.makedirs(new_src)  # generate new folders for each file

  def deleteComponents(data):
    for i in range(data.numAttributes() - 2, 0, -1):  # delete attributes from the back, except of clasl label
      if i >= number_of_componets:
        data.deleteAttributeAt(i)
    return data

  data = loadWekaData(data_src)
  search = autoclass('weka.attributeSelection.Ranker')()
  evaluator = autoclass('weka.attributeSelection.PrincipalComponents')()
  evaluator.setOptions(['-R', '0.9', '-A', '1'])
  attsel = autoclass('weka.attributeSelection.AttributeSelection')()
  attsel.setSearch(search)
  attsel.setEvaluator(evaluator)
  attsel.SelectAttributes(data)
  reduced_data = attsel.reduceDimensionality(data)
  reduced_data = deleteComponents(reduced_data)
  createfolder(data_src[2:data_src.rfind("/")])
  saver = autoclass('weka.core.converters.ArffSaver')()
  saver.setInstances(reduced_data)
  saver.setFile(autoclass("java.io.File")("./R" + data_src[2:]))
  saver.writeBatch()


def runPCA():
  datasrc = readsrc()
  for group, srclst in datasrc.iteritems():
    for one in srclst:
      PCA(one)
  return "./Rdataset"


def createfolder(new_src):
  if os.path.exists(new_src):
    return
  else:
    os.makedirs(new_src)  # generate new folders for each file


def numBuggyInstance(data):
  return len(data.attributeToDoubleArray(data.classIndex())) - sum(data.attributeToDoubleArray(data.classIndex()))
  # clean = 1, buggy:0

def small(data_src, option):
  """
  :param data_src: src of data
  :type data_src: str
  :para option: parameters for filter
  :type option: list
  """

  def selectInstanceByClass(data, num, threshold, classID):
    while num > threshold:
      index = [k for k, i in enumerate(data.attributeToDoubleArray(data.classIndex())) if i == classID]
      random_index = random.choice(index)
      data.remove(random_index)
      num = num - 1
    return data

  arff = loadWekaData(data_src)  # re-read the data
  numInstance = arff.numInstances()
  while numInstance > option[option.index("-N") + 1]:
    random_index = random.randint(0, numInstance - 1)
    arff.remove(random_index)
    numInstance -= 1
  createfolder("./Small" + data_src[2:data_src.rfind("/")])
  save(arff, "./Small" + data_src[2:])
  if "-EPV" in option  and option[option.index("-EPV") + 1] != 0:
    data = loadWekaData(data_src)  # re-read the data
    data = selectInstanceByClass(data, numBuggyInstance(data), option[option.index("-EPV") + 1], 0) # keep "-EPV" numbers of defective data
    num_instance = data.numInstances()
    if num_instance > option[option.index("-N") + 1]:
      data = selectInstanceByClass(data, data.numInstances(), option[option.index("-N") + 1], 1) # remove those additional non-defective data to get a  data set with "N" instances.
    createfolder("./EPVSmall" + data_src[2:data_src.rfind("/")])
    save(data, "./EPVSmall" + data_src[2:])


def genSmall(option):
  datasrc = readsrc()
  for group, srclst in datasrc.iteritems():
    for one in srclst:
      small(one, option)
  return "./Smalldataset"


if __name__ == "__main__":
  # read()
  # if not jvm.started: jvm.start()
  # loader = Loader(classname="weka.core.converters.ArffLoader")
  # data = loader.load_file("./dataset/AEEEM/EQ.arff")
  # data.class_is_last()
  # featureSelection(data, 9)
  # filter()
  # filter()
  PCA("./dataset/AEEEM/EQ.arff")
