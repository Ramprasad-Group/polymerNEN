"""
Copyright (C) 2019 University of Massachusetts Amherst.
This file is part of "expLinkage"
http://github.com/iesl/expLinkage
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import random
import os
import numpy as np
import torch


class Config(object):
	def __init__(self,filename=None):
		
		self.config_name 	= filename

		self.self_path      = '/path/to/repo'
		
		self.cuda  			= True
		self.useGPU 		= self.cuda and torch.cuda.is_available()
		self.seed 			= 1227
		
		
		self.mode 			= "train"
		self.resultDir 		= "/path/to/result_dir"
		self.trainResultDir = self.resultDir
		self.newDirSuffix 	= "_test_model"
		
		# File containing vectors and their cluster labels used for training or just vectors if for zero shot prediction
		self.clusterFile 	= "/path/to/repo/data/input_data/word2vec/labeled_polymer_clusters.tsv"
		self.dataDir 		= ""
		self.logFile		= "logFile.txt"
		self.trainLogFile   = self.logFile
		self.bestModel 		= "best_model"
		self.logConsole 	= True
	
		# Training Specific
		self.trainObj 		= "linkage_auto"
		self.threshold 		= 0.
		self.margin 		= 2.
		self.normalizeLoss  = False # Normalize loss for training methods other than those starting with "linkage"
		self.normExpLinkLoss  = True # Normalize loss for training methods starting with "linkage"
		self.trainExpLink 	= False
		self.scaleDist		= False # Used with VectDataTrainer and ExpLink
		self.numErrorTriplet = 1
		
		self.numEpoch 		= 100
		self.numEpToAvg		= 10
		self.epochToEval	= 1000
		self.epochToWrite	= 1000
		self.epsilon		= 0.001
		self.makeScorePlots = False
		self.evalBeforeTrain	= False
		self.evalOnTrainThresh 	= False
		self.evalOnTestThresh 	= False
		
		self.trainFrac 		= 0.75
		self.testFrac 		= 0.10
		self.devFrac 		= 0.15
		self.shuffleData 	= True

		# specific to similarity_scaling distance metric
		self.K_nn = 4
		self.use_linear_layer = True
		
		# Eval Specific
		self.thresh_scaling_factor = 2380.0*3/12802.0 # Scaling factor for the threshold used when predicting on zero-shot data
		self.inferenceMethods 	= ["singleLink@t", "avgLink@t", "compLink@t", "linkage_auto@t"]
		self.metricsForEval 	= ["f1", "randIndex"]
		
		# Scoring Model Specific Parameters
		self.modelType 		= "parameterized_cosine"
		self.inputDim 		= 128 # Dataset specific
		self.outDisSim 		= True
		self.lr 			= 0.01
		self.l2Alpha 		= 0.01
		self.alphaLr 		= 0.05
		self.alphaInitMu 	= 94
		self.alphaInitSigma = 0.01
		self.trainAlpha 	= True
		self.trainModel 	= True
		self.idenInit		= False # Useful for Mahalanobis distance learner only

		self.DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
		
		
		#if filename is not None:
		#	self.__dict__.update(json.load(open(filename)))
		
		# REDO Following three steps after updating any important parameter in config object
		self.useGPU 		= self.cuda and torch.cuda.is_available()
		self.updateRandomSeeds(self.seed)
		self.updateResultDir(self.resultDir)
		
	def to_json(self):
		return json.dumps(filter_json(self.__dict__),indent=4,sort_keys=True)

	def save_config(self, exp_dir, filename='config.json'):
		with open(os.path.join(exp_dir, filename), 'w') as fout:
			fout.write(self.to_json())
			fout.write('\n')
	
	def __getstate__(self):
		state = dict(self.__dict__)
		if "logger" in state:
			del state['logger']
			
		return state
	
	def updateResultDir(self, newResultDir):
		
		if newResultDir.startswith("auto"):
			miscInfo = newResultDir[4:]
			dataType = self.dataDir.split("/")[-1]
			self.resultDir = "{base}/d={d}/obj={obj}_s={s}{misc}".format(
				base="../results_refactor",
				d=dataType,
				obj=self.trainObj,
				s=self.seed,
				misc=miscInfo)
		else:
			self.resultDir = newResultDir
	
		
	def updateRandomSeeds(self, random_seed):
	
		self.seed = random_seed
		random.seed(random_seed)
		np.random.seed(random_seed)
		
		# self.torch_seed  = random.randint(0, 1000)
		# self.np_seed     = random.randint(0, 1000)
		self.cuda_seed   = random.randint(0, 1000)
		
		torch.manual_seed(self.seed)
		# np.random.seed(self.np_seed)
		if self.useGPU and torch.cuda.is_available():
			torch.cuda.manual_seed(self.cuda_seed)


def filter_json(the_dict):
	res = {}
	for k in the_dict.keys():
		if type(the_dict[k]) is str or \
				type(the_dict[k]) is float or \
				type(the_dict[k]) is int or \
				type(the_dict[k]) is list or \
				type(the_dict[k]) is bool or \
				the_dict[k] is None:
			res[k] = the_dict[k]
		elif type(the_dict[k]) is dict:
			res[k] = filter_json(the_dict[k])
	return res
