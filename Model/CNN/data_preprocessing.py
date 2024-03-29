import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
import torch
import json

# data1 = loadDataOrderBook('../../Data/{0}_2012-06-21_34200000_57600000_orderbook_{1}.csv', asset, level)
# data2 = loadDataMessage('../../Data/{0}_2012-06-21_34200000_57600000_message_{1}.csv', asset, level)


## load orderbook file, only allow level 1,5,10
def loadOrderBookFile (filename, asset, level):
	if level not in [1, 5, 10]:
		raise SyntaxError('Files only supports level 1, 5 and 10')
	return pd.read_csv(filename.format(asset, level))

## handler to generate column label for different level orderbook
def padLabel(level, abv):
	data_abv=abv * level
	levels = list(range(1, level + 1))
	nums = [x for x in itertools.chain.from_iterable(itertools.zip_longest(levels, levels, levels, levels)) if x]
	columns = list(map(lambda x, y: '{0}_{1}'.format(x, y), data_abv , nums))
	return columns
	
## loading orderbook data
## adjusting range of output from [0,1] to [1- result_high, result_high]
## saving result to combine_result (do we need this?)
## return X as input, Y as ground truth decision
def loadData (filename, asset, level, result_high):
	orderBook=loadOrderBookFile (filename+'_orderbook_{1}.csv', asset, level)
	message=loadOrderBookFile (filename+'_message_{1}.csv', asset, level)

	levels = list(range(1, level + 1))
	
	iters = [iter(levels), iter(levels), iter(levels), iter(levels)]
	
	nums = [x for x in itertools.chain.from_iterable(itertools.zip_longest(levels, levels, levels, levels)) if x]

	orderBook_abv = ['ask', 'volume_ask', 'bid', 'volume_bid'] * level	
	orderBook.columns = list(map(lambda x, y: '{0}_{1}'.format(x, y), orderBook_abv , nums))
	
	message_abv = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction'] 
	message.columns = message_abv
	
	cols_abv = ["ask", "volume_ask", "bid", "volume_bid"]
	col_remain=["Type", "Size", "Price", "Direction"]
	X_abv=["ask", "volume_ask", "bid", "volume_bid"]
	X_remain=["Type", "Size", "Price"]
	Y_abv=["Direction"]


	cols=padLabel(level,cols_abv)
	X_cols=padLabel(level,X_abv)
	Y_cols=["Direction"]

	cols=cols+col_remain
	X_cols=X_cols+X_remain

	#print(cols)
	#print(X_cols)
	#print(Y_cols)
	
	#Change the output format to be between 0.1 & 0.9 instead of -1 & 1
	#result_high=0.9
	result_low=1-result_high

	# using merge function by setting how='outer'
	result = orderBook.merge(message, left_index=True, right_index=True, how='left')
	#temp=result[Y_cols]
	#temp.replace({-1 : result_low, 1 : result_high}, inplace=True)
	#result[Y_cols]=temp
	
	result['Direction'].replace({-1 : result_low, 1 : result_high}, inplace=True)

	#print(result[Y_cols])
	#print("RESULTS")  
	# displaying result
	#print("Shape of the result matrix is = ",result.shape)
	#print("*****new result*****")
	#print(result.head())
	result.to_csv("combine_resutls_"+asset+"_"+str(level)+".csv", encoding='utf-8', index=False)



	result = result[cols]
	#sns.countplot(result.ask(100))
	#result.info
	# #of rows
	#print("number of rows =",result.shape[0])
	#print("number of columns =",result.shape[1])
	#print("Number of inputs for each categories")
	#result.Type.value_counts()/result.shape[0]

	#Creating the 7 Inputs for the DNN 
	X = result[X_cols]
	#print("Shape of the matrix X is:\t",X.shape)
	#print(X.head())

	#Creating the 1 output for the DNN to train the network with results 
	Y = result[Y_cols] 
	#print("Shape of the matrix Y is:\t",Y.shape,"\n")
	#print(Y.info())
	#print(Y.head())

	#return X.add_suffix('_'+asset), Y.add_suffix('_'+asset)

	return X,Y

## merge orderbook data of 2 company
## Xname and X1name only activated if there is conflicting column name.  
def mergeData(X,X1,Xname,X1name):
	X_len=min(X.shape[0],X1.shape[0])
	X = X.merge(X1, left_index=True, right_index=True, how='left',suffixes=(Xname, X1name)) 
	X=X.iloc[:X_len]

	#print("Shape of the matrix X is:\t",X.shape)
	#print(X.head())
	#print(X.tail())
	
	return X

## load multiple company
def loadMultipleData(filename,assets,level,result_high):
	for i in range(len(assets)):
		print('load asset '+ str(i))
		asset = assets[i]
		print(asset)
		[x,y]=loadData(filename, asset, level, 0.9)
		if i==0:
			X=x
			Y=y
		else:
			X=mergeData(X,x,'','_'+asset)
			Y=mergeData(Y,y,'','_'+asset)
		print('load asset '+ str(i) + ' done')
	return X,Y

	

## arrange the input by feature (ask, volume ask), instead of by company
def arrangeByFeature(X,level):
	#get suffix
	#X.filter(regex='1957$',axis=1)

	cols = ["ask", "volume_ask", "bid", "volume_bid"]
	col=["Type", "Size", "Price", "Direction"]

	cols=padLabel(level,cols)
	for i in range(len(cols)):
		col_item = cols[i]
		#get prefix
		temp1=X.filter(regex='^'+col_item+'_',axis=1)
	
		#print(i)
		#print(temp1)
		if i==0:
			temp=temp1	
		if i!=0:
			#print(temp)
			#print(temp1)
			temp=mergeData(temp,temp1,'','')

	#print('done value')
	for j in range(len(col)):
		col_item = col[j]
		#get prefix
		temp1=X.filter(regex='^'+col_item,axis=1)
		temp=mergeData(temp,temp1,'','')
	#print('done message')

	return temp


## load multiple company level 1 orderbook into 2d image
def loadMultipleData2D(filename,assets,level,result_high):
	[X,Y]=loadMultipleData(filename,assets,level,result_high)
	X = preprocess(1)(X)
	X_train_T, X_test_T, Y_train_T, Y_test_T = split_train_test_split(X, Y, 0.2)

	# dimension: number of images, image width(#row), image length (#column)
	X_train_T=torch.reshape(X_train_T, (-1,5, 7))
	X_test_T=torch.reshape(X_test_T, (-1,5, 7))
	Y_train_T=torch.reshape(Y_train_T, (-1,5, 1))
	Y_test_T=torch.reshape(Y_test_T, (-1,5, 1))

	
	return X_train_T, X_test_T, Y_train_T, Y_test_T


def loadData2D (filename, asset, level, result_high):
	[X,Y]=loadData (filename, asset, level, result_high)

	X = preprocess(1)(X)
	
	X_train_T, X_test_T, Y_train_T, Y_test_T = split_train_test_split(X, Y, 0.2)

	X_train_T=padding2D(X_train_T,level)
	X_test_T=padding2D(X_test_T,level)
	
	return X_train_T, X_test_T, Y_train_T, Y_test_T
	

def padding2D(X,level):
	X_remain=["Type", "Size", "Price"]


	X_val=X[:,:-3]
	X_message=X[:,-3:]

	#print("loading data 2d x val")
	#print(X_val)
	#print(X_val.shape)
	#print(X_message)
	#print(X_message.shape)

	X_val=torch.reshape(X_val, (-1,level, 4))
	#print(X_val)
	#print(X_val.shape)
	X_message=X_message.repeat(1,level)
	X_message=torch.reshape(X_message, (-1,level, 3))
	#print(X_message)
	#print(X_message.shape)

	X=torch.cat((X_val, X_message), 2)

	#print(X)
	#print(X.shape)

	return X


## no processing done
def f_default(X):
	return X 

## normalise with norm_x=(x-mean_x)/(std_x)
def norm(X):
	return (X-X.mean())/X.std()

## minimal scaling
def minmax(X):
	return (X-X.min())/(X.max()-X.min())

## normalise with max
def max(X):
	return X/X.max()

## log return
def log_return(X, level, num_lags):
    out_X = pd.DataFrame()
    print(out_X.head())
    for i in range(1, level + 1):
        out_X['log_return_ask_{0}'.format(i)] = np.log(X['ask_{0}'.format(i)].pct_change() + 1)

        out_X['log_return_bid_{0}'.format(i)] = np.log(X['bid_{0}'.format(i)].pct_change() + 1)

        out_X['log_ask_{0}_div_bid_{0}'.format(i)] = np.log(X['ask_{0}'.format(i)] / X['bid_{0}'.format(i)])

        out_X['log_volume_ask_{0}_div_bid_{0}'.format(i)] = np.log(X['volume_ask_{0}'.format(i)] / X['volume_bid_{0}'.format(i)])
        
        out_X['log_volume_ask_{0}'.format(i)] = np.log(X['volume_ask_{0}'.format(i)])
        
        out_X['log_volume_bid_{0}'.format(i)] = np.log(X['volume_bid_{0}'.format(i)])
        
        if i != 1:
            out_X['log_ask_{0}_div_ask_1'.format(i)] = np.log(X['ask_{0}'.format(i)] / X['ask_1'])
            out_X['log_bid_{0}_div_bid_1'.format(i)] = np.log(X['bid_{0}'.format(i)] / X['bid_1'])
            out_X['log_volume_ask_{0}_div_ask_1'.format(i)] = np.log(X['volume_ask_{0}'.format(i)] / X['volume_ask_1'])
            out_X['log_volume_bid_{0}_div_bid_1'.format(i)] = np.log(X['volume_bid_{0}'.format(i)] / X['volume_bid_1'])
        
    out_X['log_total_volume_ask'] = np.log(X[['volume_ask_{0}'.format(x) for x in list(range(1, level + 1))]].sum(axis = 1))
    out_X['log_total_volume_bid'] = np.log(X[['volume_bid_{0}'.format(x) for x in list(range(1, level + 1))]].sum(axis = 1))
            
    mid_price = (X['ask_1'] + X['bid_1']) / 2
    out_X['log_return_mid_price'] = np.log(mid_price.pct_change() + 1).shift(-1)
   
    cols_features = out_X.columns.drop(target_column)

    out_X = out_X.assign(**{
        '{}_(t-{})'.format(col, t): out_X[col].shift(t)
        for t in list(range(1, num_lags))
        for col in cols_features})

    return out_X.dropna()

## function to choose which data preprocessing to be done
def preprocess(case):
	return{
        1:norm,
		2:minmax,
		3:max,
		4:log_return,
	
	}.get(case,f_default)



def split_train_test_split(X,Y,test):
	X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(X, Y, test_size=test, shuffle = False, stratify = None)
	X_train_T = torch.from_numpy(X_train_np.to_numpy()).float()
	Y_train_T = torch.squeeze(torch.from_numpy(Y_train_np.to_numpy()).float())
	X_test_T = torch.from_numpy(X_test_np.to_numpy()).float()
	Y_test_T = torch.squeeze(torch.from_numpy(Y_test_np.to_numpy()).float())
	return X_train_T, X_test_T, Y_train_T, Y_test_T

## calculate accuracy of binary decision output
def calculate_accuracy(Y_true, Y_pred,result_high,result_low):
  predicted = Y_pred.ge(.5).view(-1) ##threshold 0.5
  return (Y_true == (predicted*(result_high-result_low)+result_low)).sum().float() / len(Y_true)

## retrieve value from tensor datatype
def round_tensor(t, decimal_places=5):
  return round(t.item(), decimal_places)



def trimming_to_7(X):
   return X[:,:7,:]


def enlargeIamge(X,N):
	print("in enlarge")
	return torch.from_numpy(np.tile(X,(1,N,N))).float()
'''
def image1Dto2D(X):
	temp=X.size(1)
	row=X.size(0)//X.size(1)
	row=row*X.size(1)
	X_temp=X[:row,:]
	X_temp=torch.unsqueeze(X_temp,0)
	return X_temp.reshape(-1,X.size(1),X.size(1))
'''	
def image1Dto2D(X):
	return X.unfold(0,X.size(1),1).transpose(2,1)

def featureChannel(X,size):
	X=X.permute(0,2,1)
	return X.unfold(0,size,1).transpose(3,2)


def displayResult(cnn_train_loss,cnn_train_acc,cnn_train_loss_q,cnn_train_acc_q,cnn_test_loss,cnn_test_acc,cnn_test_loss_q,cnn_test_acc_q):
	print('training data')
	print('cnn_train_loss')
	print(cnn_train_loss)
	print('cnn_train_acc')
	print(cnn_train_acc)
	print('cnn_train_loss_q')
	print(cnn_train_loss_q)
	print('cnn_train_acc_q')
	print(cnn_train_acc_q)


	print('testing data')
	print('cnn_test_loss')
	print(cnn_test_loss)
	print('cnn_test_acc')
	print(cnn_test_acc)
	print('cnn_test_loss_q')
	print(cnn_test_loss_q)
	print('cnn_test_acc_q')
	print(cnn_test_acc_q)
	return

def saveResult(model_num,model_struct,result_high,epoch_num,train_size,q_data_widt,learning_rate,cnn_train_loss,cnn_train_acc,cnn_train_loss_q,cnn_train_acc_q,cnn_test_loss,cnn_test_acc,cnn_test_loss_q,cnn_test_acc_q):
	filename="results case %d high %.2f %d epoch train %d %dbit"%(model_num,result_high,epoch_num,train_size,q_data_widt)+model_struct+"lr %.4f"%(learning_rate)
	np.savetxt(filename+".csv",[filename],fmt='%s')
	combined_result=np.array([cnn_train_loss,cnn_train_acc,cnn_train_loss_q,cnn_train_acc_q,cnn_test_loss,cnn_test_acc,cnn_test_loss_q,cnn_test_acc_q])
	result_df = pd.DataFrame(combined_result, columns=[filename])
	result_df.index = ["LOSS_FP_TRAIN","ACC_FP_TRAIN","LOSS_Q%d_TRAIN"%(q_data_widt),"ACC_Q%d_TRAIN"%(q_data_widt),"LOSS_FP_TEST","ACC_FP_TEST","LOSS_Q%d_TEST"%(q_data_widt),"ACC_Q%d_TEST"%(q_data_widt)]
	result_df.to_csv(filename+'.csv')

def loadData2D_featureChannel(filename,asset,level,result_high,timestep):      
	X_train_T, X_test_T, Y_train_T, Y_test_T =loadData2D(filename, asset, level, result_high)

	print("X_train_T")
	#print(X_train_T)
	print(X_train_T.shape)

    # transfer order book data to 7x10x10 block (10x10 image, with 7 channels)
	X_train_T=featureChannel(X_train_T,timestep)
	X_test_T=featureChannel(X_test_T,timestep)

    # adjust the corresponding length of Y to match X
	#Y_train_T = Y_train_T[0:-timestep+1]
	#Y_test_T = Y_test_T[0:-timestep+1]
	Y_train_T = Y_train_T[(timestep-1):]
	Y_test_T = Y_test_T[(timestep-1):]
	
	return  X_train_T, X_test_T, Y_train_T, Y_test_T

def loadMultiData2D_featureChannel(filename,assets,level,result_high,timestep):      
	for i in range(len(assets)):
		print('load asset '+ str(i))
		asset = assets[i]
		print(asset)
		[X_train_T, X_test_T, Y_train_T, Y_test_T]=loadData2D_featureChannel(filename, asset, level, result_high,timestep)
		#print(X_train_T)
		print(X_train_T.shape)
		if i==0:
			X_train=X_train_T
			X_test=X_test_T
			Y_train=Y_train_T
			Y_test=Y_test_T
		else:
			X_train_len=min(X_train.shape[0],X_train_T.shape[0])
			X_test_len=min(X_test.shape[0],X_test_T.shape[0])
			X_train=torch.cat((X_train[:X_train_len,:,:,:],X_train_T[:X_train_len,:,:,:]),dim=3)
			X_test=torch.cat((X_test[:X_test_len,:,:,:],X_test_T[:X_test_len,:,:,:]),dim=3)
		#print(X_train)
		print(X_train.shape)
		print('load asset '+ str(i) + ' done')
	return X_train, X_test, Y_train[:X_train.shape[0]], Y_test[:X_test.shape[0]]

