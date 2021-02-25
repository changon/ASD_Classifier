from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt

def write_tree(trainData, trainOutcomes, params):
	clf=DecisionTreeClassifier().fit(trainData, trainOutcomes)
	plt.rcParams["figure.figsize"] = (120, 120)
	tree.plot_tree(clf, feature_names=list(trainData.columns.values), class_names=["0", "1"], filled=True)
	plt.show()

	dtName="decision_tree"+str(params["nEpochs"])+".png"
	ffname=dtName
	print("\twriting "+ffname)
	plt.savefig(ffname)
