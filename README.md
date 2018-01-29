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
| Logistic Regression     | lr_run.py           | 66.1 %                    
| Multi Layer Perceptron  | nn_model.py         | 3.4 %  
| Conv Net + MLP       |  nn_model.py)       | 6.1 %
 
 ## Reason for model choices
 I started with a multinomial naive bayes classifier to obtain a baseline score for the dataset. I expected other models to be pretty close to the NB model given that its just a list of ingredients and there is no syntactic/semantic information being provided by the text.
