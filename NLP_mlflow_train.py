#!/usr/bin/env python
# coding: utf-8

# ## NLP Kaggle model training

# In[1]:


#breakpoint()
from tweets_emergency import *
import gc
gc.collect()


# #### Data loading for pre-processing

# In[ ]:


#setting up non-default parameters

try:
    params['threshold'] = float(sys.argv[1])
    params['lr'] = float(sys.argv[2])
except:
    pass
    
params['model'] = sys.argv[3]


# In[2]:


data = load_CSV(train_src)
data_size = len(data.index)
texts = list(data['text'])
lemmatized_texts = process_textlist(texts,
                                    lemmatize = params['lemmatize'], 
                                    remove_stop = params['remove_stop'],
                                    remove_nonalpha = params['remove_nonalpha'])
masterset = create_masterset(lemmatized_texts)


# #### Featurization

# In[3]:


vocab = create_vocabulary(masterset)
mat = create_trainmat(lemmatized_texts, vocab)
labels = np.asarray(list(data['target']), dtype = np.float32)
#np.savetxt("data/mat.csv", mat, delimiter = ",")
#np.savetxt("data/labels.csv", labels, delimiter = ",")

tweets_dataset = tweetDataset(mat, labels)
indices = list(range(data_size))
split = int(data_size * (1 - params['split']))
train_index, val_index = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_index)
val_sampler = SubsetRandomSampler(val_index)

tweets_gen_train = torch.utils.data.DataLoader(tweets_dataset, 
                                               batch_size = params['batch'], 
                                               sampler = train_sampler)

tweets_gen_val = torch.utils.data.DataLoader(tweets_dataset, 
                                             batch_size = params['batch'], 
                                             sampler = val_sampler)


# In[4]:


if __name__ == "__main__":
    
    if (params['model'] ==  'neural'):
        model = neural(len(vocab))
    else:
        model = logReg(len(vocab))
        
    conf_matrix = confusionMatrix(params['threshold'])
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = params['lr'])
    losses = []

    with mlflow.start_run() as run:
        for epoch in range(params['epochs']): # full passes over the data
            for data in tweets_gen_train:  # `data` is a batch of data
                X, y = data  # X is the batch of features, y is the batch of targets.
                model.zero_grad()  # sets gradients to 0 bairefore loss calc. You will do this likely every step.
                output = model.forward(X.float())  # forward pass
                loss = loss_function(output, y)  # calc and grab the loss value
                loss.backward()  # apply this loss backwards thru the network's parameters
                optimizer.step()  # attempt to optimize weights to account for loss/gradients
            for data in tweets_gen_val:
                X, y = data
                model.eval()
                output_hat = model.forward(X.float())
                loss_hat = loss_function(output_hat, y)
                conf_matrix.update(y, output_hat)
            #print("loss in validation set is: " + str(loss_hat.item()))  # print loss. We hope loss (a measure of wrong-ness) declines!
            losses.append(loss_hat.item())
            conf_matrix.calc_metrics()
            mlflow.log_param("epochs", params['epochs'])
            mlflow.log_param("model", params['model'])
            mlflow.log_param("learning rate", params['lr'])
            mlflow.log_param("batch size", params['batch'])
            mlflow.log_param("classification threshold", params['threshold'])
            mlflow.log_param("lemmatize", params['lemmatize'])
            mlflow.log_param("remove_stop", params['remove_stop'])
            mlflow.log_param("remove_nonalpha", params['remove_nonalpha'])
            mlflow.log_metric('accuracy', conf_matrix.accuracy, step=epoch)
            mlflow.log_metric('precision', conf_matrix.precision, step=epoch)
            mlflow.log_metric('recall', conf_matrix.recall, step=epoch)
            mlflow.log_metric('f1', conf_matrix.f1, step=epoch)
    
        #conf_matrix.output()
        #losses_np = np.asarray(losses, dtype = np.float32)
        #np.savetxt("data/losses.csv", losses_np, delimiter = ",")
        #mlflow.log_artifact("home/ubuntu/tweets_emergency/data")
        mlflow.pytorch.log_model(model, params['model'])
        


# In[5]:


for i in tweets_gen_train:
    print(i[0][1])
    print(type(i[0][1]))
    test_np = pd.DataFrame((i[0][1:5]).float())
    test_np = test_np.astype(float)
    #print(len(i[0][1]))
    break


# In[6]:


test_np


# In[7]:


model.predict(test_np)


# In[ ]:




