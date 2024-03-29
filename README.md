# Recipe Classification models

## Preprocessing
 * Reduce all words to lower case
 * Remove words in paranthesis. These are generally quantities of that ingredient. 
 * Concatenate ingredeints of each recipe (for neural network input)
 * Build feature vector (vector of word indices for neural networks and counts for the rest)
 
 ## Features
 * Bag of Words (word counts)
 * Bag of ingredients (ingredient counts) ---- Ignored due to inferior performance
 * Lookup table embedding ----- For neural network

## Models
| Model                   | Model file          | Accuracy on dev set |
|-------------------------|---------------------|---------------------|
| Multinomial Naive Bayes | baseline_methods.py | 61.6 %              |
| Linear SVM              | baseline_methods.py | 67.2 %              | 
| Logistic Regression     | lr_run.py           | 66.1 %              |      
| MLP in Keras            | keras_run.py        | 62.7% (3 epochs, batch_size= 64, lr = 0.0001)
| Multi Layer Perceptron  | nn_model.py         | 6.4 %  
| Conv Net + MLP          | nn_model.py         | 3.1 %
 
## How to run
1. Clone repo
2. Untar data files

### Run Logistic regression model (main submission):
```bash 
 cd src
 python lr_run.py
```
This runs train, validation and test. You can pass any other file to test() to test on different data

### Run Baseline methods (MNB and SVM): 
```bash
 cd src
 python baseline_methods.py
```
### Run Keras MLP model
```bash 
 cd src
 python nn_preprocessing.py  #Required only once
 python keras_run.py
``` 
### Load saved model for Keras MLP
Copy model.h5 to recipe_classification/saved_models
```bash 
 cd src
 python nn_preprocessing.py  #Required only once
 python keras_run.py 1
``` 

### Run Neural network models (Runs CNN model by default):
```bash 
 cd src
 python nn_preprocessing.py  #Required only once
 python nn_run.py
``` 

 ## Reason for model choices
 I started with a multinomial naive bayes classifier to obtain a baseline score for the dataset. I expected other models to be pretty close to the NB model given that its just a list of ingredients and there is no syntactic/semantic information being provided by the text. 
 
 With the same set of features (word counts), I tried a SVM classifier. Non linear models were taking forever to converge due to high sparsity. After some fine tuning, this model gave me the highest accuracy on dev set. However, since it uses a hinge loss, I cannot obtain a probability score as was needed in the problem.
 
 
 I then resorted to a logistic regression model to obtain a respectable classifier that also gives me probabilities. This gave me a marginally inferior score compared to SVM but gave me probabilities. I am using this model as my final submission for this Project.
 
 Neural network model is a simple 3 layer MLP for now. Although other models can be tried, I could not do it due to lack of time
Given that bag of words will be extremely sparse, I used the usual lookup based embeddings (learned) for the ingredients. I also considered pretrained word embeddings as it will be much denser representation but felt it should not give me any meaningful increase in performance. The reason being - there is no real structure in the input. It is just a set of ingredients. Were they mixed with natural language sentences, it would be more appropriate to use them as they carry syntactic and semantic information. I felt it was not needed for this task. 
 
 ## Discussion
The simple Keras MLP model itself has a lot of room for improvement. Given the time, it can get at least +5% gain in accuracy. I can also try CNN model I tried in pytorch and that is sure to give some improvement. 
 
Another possible improvement would be lemmatize/stem words in the dataset. as this would reduce the number of words in the vocabulary. But I was skeptical of this solution given that most words are nouns and the improvement would not be substantial. 
 
Features like TF-IDF scores don't help in this dataset given that it is the rare words that give any meaningful features in this dataset. 
 
The Naive Bayes classifier could be improved much more by modelling the priors better.
 
Using a decision tree or random forest classifier could also give good improvements. 

Dataset noise could be another factor. 
