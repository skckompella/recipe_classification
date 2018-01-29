# Recipe Classification models

## Preprocessing
 * Reduce all words to lower case
 * Remove words in paranthesis. These are generally quantities of that ingredient. 
 * Concatenate ingredeints of one recipe (for neural network input)
 ## Features
 * Bag of Words (word counts)
 * Bag of ingredients (ingredient counts) ---- Ignored due to inferior performance
 * Lookup table embedding ----- For neural network
