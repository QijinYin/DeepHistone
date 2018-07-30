from model import DeepHistone
import copy
import numpy as np
from utils import metrics,model_train,model_eval,model_predict
import torch
#setting 
batchsize=20
data_file = 'data/data.npz'

model_save_file = 'results/model.txt'
lab_save_file ='results/label.txt'
pred_save_file ='results/pred.txt'

print('Begin loading data...')
with np.load(data_file) as f:
	indexs = f['keys']
	dna_dict = dict(zip(f['keys'],f['dna']))
	dns_dict = dict(zip(f['keys'],f['dnase']))
	lab_dict = dict(zip(f['keys'],f['label']))
np.random.shuffle(indexs)
idx_len = len(indexs)
train_index=indexs[:int(idx_len*3/5)]
valid_index=indexs[int(idx_len*3/5):int(idx_len*4/5)]
test_index=indexs[int(idx_len*4/5):]


use_gpu = torch.cuda.is_available()
model = DeepHistone(use_gpu)
print('Begin training model...')
best_model = copy.deepcopy(model)
best_valid_auPRC=0
best_valid_loss = np.float('Inf')
for epoch in range(50):
	np.random.shuffle(train_index)
	train_loss= model_train(train_index,model,batchsize,dna_dict,dns_dict,lab_dict,)
	valid_loss,valid_lab,valid_pred= model_eval(valid_index, model,batchsize,dna_dict,dns_dict,lab_dict,)
	valid_auPRC,valid_auROC= metrics(valid_lab,valid_pred,'Valid',valid_loss)

	if np.mean(list(valid_auPRC.values())) >best_valid_auPRC:
		best_model = copy.deepcopy(model)

	if valid_loss < best_valid_loss: 
		early_stop_time = 0
		best_valid_loss = valid_loss	
	else:
		model.updateLR(0.1)
		early_stop_time += 1
		if early_stop_time >= 5: break


print('Begin predicting...')
test_lab,test_pred = model_predict(test_index,best_ model,batchsize,dna_dict,dns_dict,lab_dict,)	
test_auPR,test_roc= metrics(test_lab,test_pred,'Test')


print('Begin saving...')
np.savetxt(lab_save_file, test_lab, fmt='%d', delimiter='\t')
np.savetxt(pred_save_file, test_pred, fmt='%.4f', delimiter='\t')
best_model.save_model(model_save_file)

print('Finished.')
