from torch import nn
import torch

class classifierNetwork(nn.Module):
	def __init__(self, modelSetting):
		super(classifierNetwork, self).__init__()
		self.modelSetting=modelSetting
		inputDim=modelSetting["inputDim"]
		outDim=modelSetting["outDim"]
		nhidden=modelSetting["nhidden"]
		hiddenDims=modelSetting["hiddenDims"] #list
		
		self.dims=[inputDim]
		self.transforms=[]
		for i in range(nhidden):
			self.dims.append(hiddenDims[i])
			ti = nn.Linear(self.dims[i], self.dims[i+1])
			torch.nn.init.xavier_uniform_(ti.weight)
			self.transforms.append(ti)
		ti=nn.Linear(self.dims[nhidden], outDim)
		torch.nn.init.xavier_uniform_(ti.weight)
		self.transforms.append(ti)

		self.printStr="Network Info. \n\t nHiddenLayers: "+str(modelSetting["nhidden"])
		self.printStr+=" \n\t nHiddenDims: "+str(modelSetting["hiddenDims"])

		#above not working, so do it hard coded
		self.t1=nn.Linear(inputDim, hiddenDims[0])
		torch.nn.init.xavier_uniform_(self.t1.weight)

		self.t2=nn.Linear(hiddenDims[0], outDim)
		torch.nn.init.xavier_uniform_(self.t2.weight)

	def forward(self, x):
		#for t in self.transforms:
		#       x=t(x)
		#       x=torch.sigmoid(x)
		x=self.t1(x)
		x=torch.sigmoid(x)
		x=self.t2(x)
		x=torch.sigmoid(x)
		x=torch.sigmoid(x)
		return x

	def printMetaInfo(self):
		print(self.printStr)
