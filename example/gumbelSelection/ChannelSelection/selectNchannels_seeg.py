import argparse
import os
import random

import hdf5storage
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import models
from gesture.config import *
from comm_utils import slide_epochs
from common_dl import myDataset
from gesture.preprocess.chn_settings import get_channel_setting
from loader import within_subject_loader_HGD, all_subject_loader_HGD
from models import SelectionNet, init_weights

import statistics
from random import randint
import importlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch Channel Selection Training')

parser.add_argument('--M',type=int,default=3,

					help='number of selection neurons')

parser.add_argument('--epochs',type=int, default=200,

					help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', type=int, default=16, 

					help='mini-batch size')

parser.add_argument('--gradacc', type = int, default=1,

					help='gradient accumulation')

parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4, 

					help='weight decay')

parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, 

					help='initial learning rate')

parser.add_argument('--lamba', type=float, default=0.1, 

					help='regularization weight')

parser.add_argument('--start_temp',type=float,default=10.0,

					help='initial temperature')
parser.add_argument('--end_temp',type=float,default=0.1,

					help='final temperature')

parser.add_argument('--train_split',type=float,default=0.8,

					help='training-validation data split')
parser.add_argument('--patience', type=int, default=10,

					help='amount of epochs before early stopping')

parser.add_argument('--stop_delta', type=float, default=1e-3,

					help='maximal drop in validation loss for early stopping')

parser.add_argument('--entropy_lim', type=float, default=0.05,

					help='mean entropy for the selection neurons to be reached for convergence')

parser.add_argument('--seed',type=int,default=0,

					help='random seed, 0 indicates randomly chosen seed')

parser.add_argument('-v', action="store_true", default=True, dest="verbose")


def main():
	############## seeg data ##########
	sid = 10  # 4
	fs=1000
	class_number = 5
	Session_num, UseChn, EmgChn, TrigChn, activeChan = get_channel_setting(sid)

	loadPath = data_dir + 'preprocessing' + '/P' + str(sid) + '/preprocessing2.mat'
	mat = hdf5storage.loadmat(loadPath)
	data = mat['Datacell']
	channelNum = int(mat['channelNum'][0, 0])
	data = np.concatenate((data[0, 0], data[0, 1]), 0)
	del mat
	# standardization
	# no effect. why?
	if 1 == 1:
		chn_data = data[:, -3:]
		data = data[:, :-3]
		scaler = StandardScaler()
		scaler.fit(data)
		data = scaler.transform((data))
		data = np.concatenate((data, chn_data), axis=1)

	# stim0 is trigger channel, stim1 is trigger position calculated from EMG signal.
	chn_names = np.append(["seeg"] * len(UseChn), ["stim0", "emg", "stim1"])
	chn_types = np.append(["seeg"] * len(UseChn), ["stim", "emg", "stim"])
	info = mne.create_info(ch_names=list(chn_names), ch_types=list(chn_types), sfreq=fs)
	raw = mne.io.RawArray(data.transpose(), info)

	# gesture/events type: 1,2,3,4,5
	events0 = mne.find_events(raw, stim_channel='stim0')
	events1 = mne.find_events(raw, stim_channel='stim1')
	# events number should start from 0: 0,1,2,3,4, instead of 1,2,3,4,5
	events0 = events0 - [0, 0, 1]
	events1 = events1 - [0, 0, 1]

	# print(events[:5])  # show the first 5
	# Epoch from 4s before(idle) until 4s after(movement) stim1.
	raw = raw.pick(["seeg"])
	epochs = mne.Epochs(raw, events1, tmin=0, tmax=4, baseline=None)
	# or epoch from 0s to 4s which only contain movement data.
	# epochs = mne.Epochs(raw, events1, tmin=0, tmax=4,baseline=None)

	epoch1 = epochs['0'].get_data()  # 20 trials. 8001 time points per trial for 8s.
	epoch2 = epochs['1'].get_data()
	epoch3 = epochs['2'].get_data()
	epoch4 = epochs['3'].get_data()
	epoch5 = epochs['4'].get_data()
	list_of_epochs = [epoch1, epoch2, epoch3, epoch4, epoch5]
	total_len = list_of_epochs[0].shape[2]

	# validate=test=2 trials
	trial_number = [list(range(epochi.shape[0])) for epochi in list_of_epochs]  # [ [0,1,2,...19],[0,1,2...19],... ]
	test_trials = [random.sample(epochi, 2) for epochi in trial_number]
	# len(test_trials[0]) # test trials number
	trial_number_left = [np.setdiff1d(trial_number[i], test_trials[i]) for i in range(class_number)]

	val_trials = [random.sample(list(epochi), 2) for epochi in trial_number_left]
	train_trials = [np.setdiff1d(trial_number_left[i], val_trials[i]).tolist() for i in range(class_number)]

	# no missing trials
	assert [sorted(test_trials[i] + val_trials[i] + train_trials[i]) for i in range(class_number)] == trial_number

	test_epochs = [epochi[test_trials[clas], :, :] for clas, epochi in
				   enumerate(list_of_epochs)]  # [ epoch0,epoch1,epch2,epoch3,epoch4 ]
	val_epochs = [epochi[val_trials[clas], :, :] for clas, epochi in enumerate(list_of_epochs)]
	train_epochs = [epochi[train_trials[clas], :, :] for clas, epochi in enumerate(list_of_epochs)]

	wind = 500
	stride = 500
	X_train = []
	y_train = []
	X_val = []
	y_val = []
	X_test = []
	y_test = []

	for clas, epochi in enumerate(test_epochs):
		Xi, y = slide_epochs(epochi, clas, wind, stride)
		assert Xi.shape[0] == len(y)
		X_test.append(Xi)
		y_test.append(y)
	X_test = np.concatenate(X_test, axis=0)  # (1300, 63, 500)
	y_test = np.asarray(y_test)
	y_test = np.reshape(y_test, (-1, 1))  # (5, 270)

	for clas, epochi in enumerate(val_epochs):
		Xi, y = slide_epochs(epochi, clas, wind, stride)
		assert Xi.shape[0] == len(y)
		X_val.append(Xi)
		y_val.append(y)
	X_val = np.concatenate(X_val, axis=0)  # (1300, 63, 500)
	y_val = np.asarray(y_val)
	y_val = np.reshape(y_val, (-1, 1))  # (5, 270)

	for clas, epochi in enumerate(train_epochs):
		Xi, y = slide_epochs(epochi, clas, wind, stride)
		assert Xi.shape[0] == len(y)
		X_train.append(Xi)
		y_train.append(y)
	X_train = np.concatenate(X_train, axis=0)  # (1300, 63, 500)
	y_train = np.asarray(y_train)
	y_train = np.reshape(y_train, (-1, 1))  # (5, 270)
	chn_num = X_train.shape[1]

	train_set = myDataset(X_train, y_train)
	val_set = myDataset(X_val, y_val)
	test_set = myDataset(X_test, y_test)

	batch_size = 32
	train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=False)
	val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=False)
	test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, pin_memory=False)

	########## end  seeg #########################
	global args,enable_cuda
################################################################ INIT #################################################################################
	
	args = parser.parse_args()

	cwd=os.getcwd()
	#dpath=os.path.dirname(cwd)
	dpath='/Volumes/Samsung_T5/data/braindecode/'
	result_dir=dpath+'result/'
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	#Paths for data, model and checkpoint
	data_path = os.path.join(dpath,'Data/')
	model_save_path = os.path.join(dpath,'Models','Model_GumbelregHighgamma_M'+str(args.M))
	checkpoint_path = os.path.join(dpath,'Models','Checkpoint_GumbelregHighgamma_M'+str(args.M))
	if not os.path.isdir(os.path.join(dpath,'Models')):
		os.makedirs(os.path.join(dpath,'Models'))

	#Check if CUDA is available
	enable_cuda = torch.cuda.is_available()
	if(args.verbose):
		print('GPU computing: ', enable_cuda)

	#Set random seed
	if(args.seed==0):
		args.seed=randint(1,99999)

	#Initialize devices with random seed
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	training_accs = []
	val_accs=[]
	test_accs = []

	#Create a vector of length epochs, decaying start_value to end_value exponentially, reaching end_value at end_epoch
	def exponential_decay_schedule(start_value,end_value,epochs,end_epoch):
		t = torch.FloatTensor(torch.arange(0.0,epochs))
		p = torch.clamp(t/end_epoch,0,1)
		out = start_value*torch.pow(end_value/start_value,p)

		return out

	#Network loss function
	def loss_function(output,target,model,lamba,weight_decay):
		l = nn.CrossEntropyLoss()
		sup_loss = l(output,target)
		reg = model.regularizer(lamba,weight_decay)

		return sup_loss,reg

	#Create schedule for temperature and regularization threshold
	temperature_schedule = exponential_decay_schedule(args.start_temp,args.end_temp,args.epochs,int(args.epochs*3/4))
	thresh_schedule = exponential_decay_schedule(10.0,1.1,args.epochs,args.epochs)

	#Load data
	num_subjects = 5
	input_dim=[44,1125]
	#train_loader1,val_loader1,test_loader1 = all_subject_loader_HGD(batch_size=args.batch_size,train_split=args.train_split,path=data_path,num_subjects=num_subjects)


################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################

	if(args.verbose):
		print('Start training')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	#Instantiate model
	model = SelectionNet(input_dim,args.M).float()
	if(enable_cuda):
		model.cuda()
	model.set_freeze(False)

	optimizer = torch.optim.Adam(model.parameters(),args.lr)

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0
	fig, ax=plt.subplots()

	while epoch in range(args.epochs) and (not early_stop):

		#Update temperature and threshold
		model.set_thresh(thresh_schedule[epoch])
		model.set_temperature(temperature_schedule[epoch])

		#Perform training step
		train(train_loader, model, loss_function, optimizer,epoch,args.weight_decay,args.lamba,args.gradacc,args.verbose)
		val_loss = validate(val_loader,model,loss_function,epoch,args.weight_decay,args.lamba,args.verbose)
		tr_acc,val_acc,test_acc=test(train_loader,val_loader,test_loader,model,loss_function,args.weight_decay,args.verbose)

		#Extract selection neuron entropies, current selections and probability distributions
		H,sel,probas = model.monitor()
		ax.plot(probas.detach().numpy())
		fig.savefig(result_dir+'prob_dist'+str(epoch)+'.png')
		ax.clear()
		#fig.clear()

		#If selection convergence is reached, enable early stopping scheme
		if((torch.mean(H.data)<=args.entropy_lim) and (val_loss>prev_val_loss-args.stop_delta)):
			patience_timer+=1
			if(args.verbose):
				print('Early stopping timer ', patience_timer)
			if(patience_timer == args.patience):
				early_stop = True
		else:
			patience_timer=0
			H,sel,probas = model.monitor()
			torch.save(model.state_dict(),checkpoint_path)
			prev_val_loss = val_loss


		epoch+=1

	if(args.verbose):
		print('Channel selection finished')

	#Store subject independent model
	model.load_state_dict(torch.load(checkpoint_path))
	pretrained_path = str(model_save_path+'all_subjects_channels_selected.pt')
	torch.save(model.state_dict(), pretrained_path) 

################################################################ SUBJECT FINETUNING  #################################################################################
## freeze the selection layer and train the model other part
	if(args.verbose):
		print('Start subject specific training')

	for k in range(1,num_subjects+1):


		if(args.verbose):
			print('Start training for subject ' + str(k))

		torch.manual_seed(args.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

		#Load subject independent model and freeze selection neurons
		model = SelectionNet(input_dim,args.M)
		model.load_state_dict(torch.load(pretrained_path))
		if(enable_cuda):
			model.cuda()
		model.set_freeze(True)

		#Load subject dependent data
		train_loader,val_loader,test_loader = within_subject_loader_HGD(subject=k,batch_size=args.batch_size,train_split=args.train_split,path=data_path)
	
		optimizer = torch.optim.Adam(model.parameters(),args.lr)

		prev_val_loss = 100
		patience_timer = 0
		early_stop = False
		epoch = 0
		while epoch in range(args.epochs) and (not early_stop):

			#Perform train step
			train(train_loader, model, loss_function, optimizer,epoch,args.weight_decay,args.lamba,args.gradacc,args.verbose)
			val_loss = validate(val_loader,model,loss_function,epoch,args.weight_decay,args.lamba,args.verbose)
			tr_acc,val_acc,test_acc=test(train_loader,val_loader,test_loader,model,loss_function,args.weight_decay,args.verbose)

			#Extract selection neuron entropies, current selections and probability distributions   
			H,sel,probas = model.monitor()

			#Perform early stopping
			if(val_loss>prev_val_loss-args.stop_delta):
				patience_timer+=1
				if(args.verbose):
					print('Early stopping timer ', patience_timer)
				if(patience_timer == args.patience):
					early_stop = True
			else:
				patience_timer=0
				torch.save(model.state_dict(),checkpoint_path)
				prev_val_loss = val_loss

			epoch+=1


		#Store model with lowest validation loss
		model.load_state_dict(torch.load(checkpoint_path))    
		path = str(model_save_path+'finished_subject'+str(k)+'.pt')
		torch.save(model.state_dict(), path)
			
		#Evaluate model
		tr_acc,val_acc,test_acc = test(train_loader,val_loader,test_loader,model,loss_function,args.weight_decay,args.verbose)
		training_accs.append(tr_acc)
		val_accs.append(val_acc)
		test_accs.append(test_acc)

 ################################################################ TERMINATION  #################################################################################           

	print('Selection', sel.data)
	print('Training accuracies', training_accs)
	print('Validation accuracies', val_accs)
	print('Testing accuracies', test_accs)

	tr_med = statistics.median(training_accs)
	val_med = statistics.median(val_accs)
	test_med = statistics.median(test_accs)
	tr_mean = statistics.mean(training_accs)
	val_mean = statistics.mean(val_accs)
	test_mean = statistics.mean(test_accs)

	print('Training median accuracy', tr_med)
	print('Validation median accuracy', val_med)
	print('Testing median accuracy', test_med)
	print('Training mean accuracy', tr_mean)
	print('Validation mean accuracy', val_mean)
	print('Testing mean accuracy', test_mean)

#train 1 epoch
def train(train_loader, model, loss_function, optimizer, epoch, weight_decay,lamba,gradacc,verbose):

	global running_loss, running_sup_loss, running_reg, running_acc,enable_cuda

	model.train()

	for i, (data, labels) in enumerate(train_loader):

		if(enable_cuda):
			data= data.cuda().float()
			labels = labels.cuda()

		if(i==0):
			running_loss = 0.0
			running_reg = 0.0
			running_sup_loss = 0.0
			running_acc = np.array([0,0])

		output = model(data) # shape: torch.Size([16, 1, 44, 1125])

		sup_loss,reg = loss_function(output, labels, model,lamba,weight_decay)
		loss = sup_loss + reg # regulization
		loss=loss/gradacc

		loss.backward()

		#Perform gradient accumulation
		if((i+1)%gradacc ==0):
			optimizer.step()
			optimizer.zero_grad()

		#running accuracy
		score, predicted = torch.max(output,1)
		total = predicted.size(0)
		correct = (predicted == labels).sum().item()
		running_acc = np.add(running_acc, np.array([correct,total]))

		# print statistics
		running_loss += loss.item()
		running_reg += reg.item()
		running_sup_loss += sup_loss.item()
		N = len(train_loader)
		if(i==N-1):
			if(verbose):
				print('[%d, %5d] loss: %.3f acc: %d %% supervised loss: %.3f regularization loss %.3f'%
						(epoch + 1, i + 1, running_loss / N, 100*running_acc[0]/running_acc[1], running_sup_loss/N, running_reg/N))
			running_loss = 0.0
			running_reg = 0.0
			running_sup_loss = 0.0
			running_acc = (0,0)

def validate(val_loader,model,loss_function,epoch,weight_decay,lamba,verbose):

	global val_acc,val_loss,enable_cuda

	with torch.no_grad():
		model.eval()

		for i, (data, labels) in enumerate(val_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			if(i==0):
				val_loss = 0.0
				val_acc = np.array([0,0])

			output = model(data)
			sup_loss,reg = loss_function(output, labels, model,lamba,weight_decay)
			loss = sup_loss

			#running accuracy
			score, predicted = torch.max(output,1)
			total = predicted.size(0)
			correct = (predicted == labels).sum().item()
			val_acc = np.add(val_acc, np.array([correct,total]))

			# print statistics
			val_loss += loss.item()
			N = len(val_loader)
			if(i == N-1):
				if(verbose):
					print('[%d, %5d] Validation loss: %.3f Validation accuracy: %d %%'%
						(epoch + 1, i + 1, val_loss / N,100*val_acc[0]/val_acc[1] ))

	return val_loss/N

def test(train_loader,val_loader,test_loader, model,loss_function,weight_decay,verbose):

	global enable_cuda

	with torch.no_grad():

		model.train()

		total = 0
		correct = 0

		for i, (data, labels) in enumerate(train_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			output = model(data)
			score, predicted = torch.max(output,1)
			total += predicted.size(0)
			correct += (predicted == labels).sum().item()

		tr_acc = correct/total

		if(verbose):
			print('Training accuracy: %d %%' % (100 * tr_acc))

		model.eval()

		total = 0
		correct = 0

		for i, (data, labels) in enumerate(val_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			output = model(data)
			score, predicted = torch.max(output,1)
			total += predicted.size(0)
			correct += (predicted == labels).sum().item()

		val_acc = correct/total

		if(verbose):
			print('Validation accuracy: %d %%' % (100 * val_acc))

		total=0
		correct=0

		for i, (data, labels) in enumerate(test_loader):

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			output = model(data)
			score, predicted = torch.max(output,1)
			total += predicted.size(0)
			correct += (predicted == labels).sum().item()

		test_acc = correct/total

		if(verbose):
			print('Test accuracy: %d %%' % (100 * test_acc))

		return tr_acc,val_acc,test_acc

if __name__ == '__main__':

	main()
