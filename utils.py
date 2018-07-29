from sklearn.metrics import auc,roc_auc_score,precision_recall_curve
import numpy as np
histones=['H3K4me1','H3K4me3','H3K27me3','H3K36me3','H3K9me3','H3K9ac','H3K27ac']

def loadRegions(regions_indexs,dna_dict,dns_dict,label_dict,):
	if dna_dict is not None:
		dna_regions = np.concatenate([dna_dict[meta]  for meta in regions_indexs],axis=0)
	else: dna_regions =[]
	if dns_dict is not None:
		dns_regions = np.concatenate([dns_dict[meta] for meta in regions_indexs],axis=0)
	else: dns_regions =[]
	label_regions = np.concatenate([label_dict[meta] for meta in regions_indexs],axis=0).astype(int)
	return dna_regions,dns_regions,label_regions
 	
def model_train(regions,model,batchsize,dna_dict,dns_dict,label_dict,):
	train_loss = []
	regions_len = len(regions)
	for i in range(0, regions_len , batchsize):
		regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
		seq_batch ,dns_batch,lab_batch = loadRegions(regions_batch,dna_dict,dns_dict,label_dict)
		_loss= model.train_on_batch(seq_batch, dns_batch, lab_batch)
		train_loss.append(_loss)
	return np.mean(train_loss) 

def model_eval(regions,model,batchsize,dna_dict,dns_dict,label_dict,):
	loss = []
	pred =[]
	lab =[]
	regions_len = len(regions)
	for i in range(0, regions_len , batchsize):
		regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
		seq_batch ,dns_batch,lab_batch = loadRegions(regions_batch,dna_dict,dns_dict,label_dict)
		_loss,_pred = model.eval_on_batch(seq_batch, dns_batch, lab_batch)
		loss.append(_loss)
		lab.extend(lab_batch)
		pred.extend(_pred)
	return np.mean(loss), np.array(lab),np.array(pred)

def model_predict(regions,model,batchsize,dna_dict,dns_dict,label_dict,):
	lab  = []
	pred = []
	regions_len = len(regions)
	for i in range(0, len(regions), batchsize):
		regions_batch = [regions[i+j] for j in range(batchsize) if (i+j) < regions_len]
		seq_batch ,dns_batch,lab_batch = loadRegions(regions_batch,dna_dict,dns_dict,label_dict)
		_pred = model.test_on_batch(seq_batch, dns_batch)
		lab.extend(lab_batch)
		pred.extend(_pred)		
	return np.array(lab), np.array(pred) 


def ROC(label,pred):
	if len(np.unique(np.array(label).reshape(-1)))  == 1:
		print("all the label are the same !")
		return 0
	else:
		label = np.array(label).reshape(-1)
		pred = np.array(pred).reshape(-1)
		return roc_auc_score(label,pred)
def auPR(label,pred):
	if len(np.unique(np.array(label).reshape(-1)))  == 1:
		print("all the label are the same !")
		return 0
	else:
		label = np.array(label).reshape(-1)
		pred = np.array(pred).reshape(-1)
		precision, recall, thresholds = precision_recall_curve(label,pred)
		return auc(recall,precision)
def metrics(lab,pred,Type='test',loss=None):
		if Type == 'Valid':
			training_color = '\033[0;34m'
		elif Type == 'Test':
			training_color = '\033[0;35m'
		else:
			training_color = '\033[0;36m'

		auPRC_dict={}
		auROC_dict ={}
		for i in range(len(histones)):
			auPRC_dict[histones[i]] = auPR(lab[:,i],pred[:,i])
			auROC_dict[histones[i]] = ROC(lab[:,i],pred[:,i])

		print_str = training_color + '\t%s\t%s\tauROC : %.4f,auPRC : %.4f\033[0m'
		print('-'*25+Type+'-'*25)
		if loss is not None: loss_str = ',Loss : %.4f'%loss
		else :loss_str =''
		print('\033[0;36m%s\tTotalMean\tauROC : %.4f,auPRC : %.4f%s\033[0m'%(Type,np.mean(list(auROC_dict.values())),np.mean(list(auPRC_dict.values())),loss_str) )
		for histone in histones:
				print(print_str%(Type,histone.ljust(10),auROC_dict[histone],auPRC_dict[histone]))
		return auPRC_dict,auROC_dict