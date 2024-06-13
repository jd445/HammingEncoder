"""
This file uses pomegranate's implementation of HMM to perform supervised multi-goal classification.
We learn a separate HMM (unsupervised) for traces from each label, and classify by choosing the HMM with highest
probability. 
"""


from random import shuffle 
from pomegranate import *
import numpy as np 
from random import *

VALIDATION = True
TREE_MINERROR = True
MULTILABEL = False

def generate_train_test_splits(input_dir,base_file,n_instances):

	data = open(input_dir + base_file).read().strip().split("\n")
	traces_by_goal = {}

	for line in data:
		t,g = line.split(";")

		if g not in traces_by_goal:
			traces_by_goal[g] = []
		traces_by_goal[g].append(line)

	for file_i in range(n_instances):
		train_traces = []
		test_traces = []

		for g in traces_by_goal:
			shuffle(traces_by_goal[g])

			n_test = max(1, int(0.2 * len(traces_by_goal[g])))

			for i in range(n_test):
				test_traces.append(traces_by_goal[g][i])

			for i in range(n_test, len(traces_by_goal[g])):
				train_traces.append(traces_by_goal[g][i])

		shuffle(train_traces)
		shuffle(test_traces)

		train_outfile = open(input_dir + "train_" + str(file_i+1) + ".txt", "w")
		test_outfile = open(input_dir + "test_" + str(file_i+1) + ".txt", "w")

		for trace in train_traces:
			train_outfile.write(trace + "\n")

		for trace in test_traces:
			test_outfile.write(trace + "\n")

		train_outfile.close()
		test_outfile.close()
  
  
class HMM():
	def __init__(self, base, iteration):
		self.base = base
		self.iteration = iteration

		train_file = "traces/" + base + "/train_" + str(iteration) + ".txt"
		test_file = "traces/" + base + "/test_" + str(iteration) + ".txt"
		self.train_traces_by_goal = {}

		self.validation_traces = []
		self.validation_labels = []

		self.test_traces = []
		self.test_labels = []
	
		self.count = 0
		self.parse_train_test(train_file, test_file)

		# We have one model for each possible goal/label
		self.models = {} 

	def parse_train_test(self, train_file, test_file):
		train_data = open(train_file).read().strip().split("\n")
		test_data = open(test_file).read().strip().split("\n")

		for line in train_data:
			t,g = line.split(";")
			t = t.strip().split(",")

			if g in self.train_traces_by_goal:
				self.train_traces_by_goal[g].append(np.array(t))
			else:
				self.train_traces_by_goal[g] = [np.array(t)]

		for goal in self.train_traces_by_goal:
			if VALIDATION:
				n_validation = max(int(len(self.train_traces_by_goal[goal]) * 0.2), 1)
			else:
				n_validation = 0
			for i in range(n_validation):
				self.validation_traces.append(self.train_traces_by_goal[goal][i])
				self.validation_labels.append(goal)

			self.train_traces_by_goal[goal] = self.train_traces_by_goal[goal][n_validation:]


		for line in test_data:
			t,g = line.split(";")
			t = t.strip().split(",")

			self.test_traces.append(np.array(t))
			self.test_labels.append(g)

	def train_hmms(self):
		best_validation_acc = -1

		for n_components in [5, 10]:
			for pseudocount in [0, 0.1, 1]:
				models = {}

				for g in self.train_traces_by_goal:
					print(g, len(self.train_traces_by_goal[g]))
					X = self.train_traces_by_goal[g]
					
					models[g] = HiddenMarkovModel.from_samples(distribution=DiscreteDistribution, n_components=n_components, X=X, algorithm='baum-welch', pseudocount=pseudocount, verbose=False, stop_threshold=1e-3, max_iterations=1e6)
					models[g].bake(merge="None")

				validation_acc = self.validation(models)
				
				if validation_acc > best_validation_acc:
					best_validation_acc = validation_acc
					self.models = models

	def train_hmms_no_validation(self):

		n_components = 10
		pseudocount = 1

		for g in self.train_traces_by_goal:
			print(g, len(self.train_traces_by_goal[g]))
			X = self.train_traces_by_goal[g]
			
			self.models[g] = HiddenMarkovModel.from_samples(distribution=DiscreteDistribution, n_components=n_components, X=X, algorithm='baum-welch', pseudocount=pseudocount, verbose=False, stop_threshold=1e-3, max_iterations=1e6)
			self.models[g].bake(merge="None")

	def validation(self, models):
		acc = 0 
		for (tau, goal) in zip(self.validation_traces, self.validation_labels):
			acc += self.eval_trace(tau, goal, models)

		return acc / len(self.validation_traces)


	## We need to manually transform the sequence because of a bug in pomegranate
	def transform_seq(self, seq, keymap):
		output = numpy.empty(len(seq), dtype=numpy.float64)
		for i in range(len(seq)):
			if seq[i] in keymap:
				output[i] = keymap[seq[i]]
			else:
				output[i] = -1
		return output 

	def eval_trace(self, tau, g_true, models = None):
		max_logp = -100000000000
		g_pred = None 

		if models is None:
			models = self.models

		for g in self.models:
			keymap = models[g].keymap[0]
			logp = models[g].log_probability(self.transform_seq(tau, keymap), check_input=False) 
			# print(g, logp)
			if logp > max_logp:
				max_logp = logp 
				g_pred = g

		if g_pred == g_true:
			return 1
		else:
			return 0

	def evaluate(self,times, dataset, output_to_file = True):
		max_len = max([len(tau) for tau in self.test_traces])

		percent_acc = [0]
		pcts = [1]
		
		for (tau, goal) in zip(self.test_traces, self.test_labels):

			## Percent accuracy
			for i in range(len(pcts)):
				length = max(1, int(pcts[i] * len(tau)))
				result = self.eval_trace(tau[:length], goal)
				percent_acc[i] += result

		for i in range(len(percent_acc)):
			percent_acc[i] /= len(self.test_traces)

		print(", ".join(map(str,percent_acc)))

		if output_to_file:
			outf = open("traces/" + dataset +f"/results/{times}hmm_" + str(iteration) + ".txt", "w")
			outf.write(", ".join(map(str,percent_acc)))

		return percent_acc



import pandas as pd

datasets = ['aslbu', 'auslan2', 'context', 'epitope', 'gene', 'pioneer', 'question', 'reuters', 'robot', 'skating', 'unix']
# datasets = ['activity']

for dataset in datasets:
	acc_all = []
	for times in range(10):
		input_dir = "traces/{}/".format(dataset)
		base_file = dataset+'_dfa.txt'
		print(base_file)
		n_instances = 5
		generate_train_test_splits(input_dir, base_file, n_instances)
		acc_5 = []
		for iteration in range(1,6):
			hmm_model = HMM(dataset, iteration)
			hmm_model.train_hmms()
			acc = hmm_model.evaluate(str(times), dataset)
			acc_5.append(acc)
		# mean on acc_5
		acc_5_mean = np.array(acc_5).mean(axis=0)
#    mean and std of 5 iterations and save to file
		acc_all.append(acc_5_mean)
	acc_all = np.array(acc_all)
	result = pd.DataFrame(acc_all.mean(axis=0), acc_all.std(axis=0))
#    /mnt/c/Users/14153/OneDrive - mail.dlut.edu.cn/DISC-master/results
	result.to_csv("results/{}_hmm.csv".format(dataset), index=False)
