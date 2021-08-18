
import torch
from torch.autograd import Variable
import torch.nn.functional as f
import numpy as np

class ParameterizedCosine(torch.nn.Module):
	
	def __init__(self,config):
		super(ParameterizedCosine, self).__init__()
		
		self.config = config
		self.clf = None
		self.inputDim = config.inputDim
		self.outputDim = self.inputDim
		self.seqModel = torch.nn.Sequential(
          torch.nn.Linear(self.inputDim,self.outputDim, bias=False)
        )
		tempAlphaVal = np.random.normal(self.config.alphaInitMu, self.config.alphaInitSigma, 1)[0]
		if self.config.useGPU:
			self.linkAlpha = Variable(torch.cuda.FloatTensor([tempAlphaVal]), requires_grad=True)
		else:
			self.linkAlpha = Variable(torch.FloatTensor([tempAlphaVal]), requires_grad=True)

		if config.idenInit: # Initialize with Identity Matrix
			self.seqModel[0].weight.requires_grad = False
			self.seqModel[0].weight.data = torch.eye(self.inputDim)
			self.seqModel[0].weight.requires_grad = True

	def __str__(self):
		printStr = ""
		printStr += "-----------------Classifier Parameters-----------------------------" + "\n"
		printStr += str(self.clf)
		printStr += "-------------------------------------------------------------------"
		return printStr
	
	def getWeightStr(self):
		return "\n\nNo parameters\n\n"
	
	def pairForward(self, pairFeature):
		raise NotImplementedError
		# prediction = self.clf.predict(pairFeature)
		# return torch.autograd.Variable(torch.FloatTensor(prediction),requires_grad=False)
	
	def pairBatchForward(self, pairFeatureList):
		prediction = self.clf.predict(pairFeatureList)
		prediction = torch.FloatTensor(prediction).view(-1,1)
		return torch.autograd.Variable(prediction, requires_grad=False)
	
	def forward(self, point1, point2):
		raise NotImplementedError
	
	# This function does not return a pytorch Variable.
	# Just the distance between point1 and point2 as per current model
	def forwardPlain(self, point1, point2):
		raise NotImplementedError

	def transformPoints(self, pointList):

		if self.config.useGPU:
			pointList = torch.cuda.FloatTensor(pointList)
		else:
			pointList = torch.Tensor(pointList)
		transformedPointList = self.seqModel(pointList)
		if self.config.useGPU:
			transformedPointList   = transformedPointList.cpu().data.numpy()
		else:
			transformedPointList = transformedPointList.data.numpy()

		return transformedPointList

	# Takes list of points and returns an adjacency matrix for it of size n x n
	def batchForwardWithin(self, points):
		numPoints = len(points)
		if self.config.useGPU:
			pointList1 = torch.cuda.FloatTensor(points)
			#pointList2 = torch.cuda.FloatTensor(points)
		else:
			pointList1 = torch.Tensor(points)
			#pointList2 = torch.Tensor(points)

		embedList = self.seqModel(pointList1).view(numPoints, self.outputDim) # Apply a linear layer to the embedding matrix
		embedListNormalized = f.normalize(embedList, p=2, dim=1)
		#embedList2 = self.seqModel(pointList2).view(1, numPoints, self.outputDim)

		# Use broadcasting feature to get nXn matrix where (i,j) contains ||p_i - p_j||_2
		ones_distance = torch.ones(numPoints, numPoints)
		distMatrix = torch.add(ones_distance, - torch.matmul(embedListNormalized, embedListNormalized.t()).view(numPoints, numPoints))

		return distMatrix
	
	# Takes list of 2 points and returns an adjacency matrix for them of size n1 x n2
	def batchForwardAcross(self, pointList1, pointList2):
		raise NotImplementedError
	
	def batchForwardOneToOne(self, pointList1, pointList2):
		raise NotImplementedError
		


if __name__ == '__main__':
	torch.manual_seed(2)
	np.random.seed(1)
	print("There is no code to run here...")