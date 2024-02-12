# Supportiv_assignment
## Solution to Relevance Sentence Pair Classification:

Let's analyze the training data:
```
Total pairs = 4000
Relevant pairs = ~1k
Non-Relevant pairs = ~3k
```
There is class imbalance here, will come to this later.

### Approach
After going through the data, to correctly pair-up the sentences, the model needs to capture the meaning and not just the context in the given 4000 pairs. So chosen DistilBert <br>

#### First how is the input modelled? <br>
input sentence = Text1 + ' [SEP] ' + Text2 <br>
label = 0 (if non-relevant) 1 (if relevant) <br>
#### Model Summary
Model:
Input is passed through the Distilbert and we take the zero-position embedding([CLS]), let it be X(dim=768)
we project this 768-d X to 768-d again => A non-linear activation => Dropout (with probability 0.2) => project to 2-d(logits) <br>
Loss is CrossEntropyLoss <br>
Inference going through the same steps

#### Why not a similarity based model?
Out of my own interest first i tried to model softmax(dot-product(BERT(sentence1),BERT(sentence2))) but quickly realized this is not generalizable and didn't work too, think of fully non-relevant pairs(3*3, no 3 pairs can be non-relevant to each other) so similarity models will tend to have limitation

#### Why input is Text1 + ' [SEP] ' + Text2
The intuition i have is we need tokens of Text1 to attend to Text2 and vice versa and determine the relevance, and they also have to attend to tokens of the same sentence(self-attention and cross-attention) => a word attends to every word in 2 sentences so we can make a sequence and let the model learn the correct interpretations and attentions. [SEP] here is a seperator.

#### why DistilBert and not Bert?

Right, overfitting problem.
After first starting with BERT, the model is overfitting to training data very quickly but no improvement in the validation/test metrics. I have tried to regularize through increasing the dropout in BERT, in projector network, weight decay but no improvement was shown. I have chosen less capacity but equally efficient model and hence chosen DistilBert for the need of the task

#### Now the Imbalance Problem
If we simply output 0, we get 75% accuracy so this is not relevant metric. Since we are trying to match relevant sentences, we need to focus on precision and recall of relevant sentence predictions.
With the same data, I was able to achieve high 95s of accuracy on training but only high 75s on validation set. 75 isn't high as we have discussed already. The precision and recall are at <0.6. First tried weight based sampling in dataloader, tried DICE loss but there isn't much improvement. Finally augmented data of relevant sentences class.

By bruteforce we can take lemmas from wordnet synsets but i used nlpaug library that is very sophisticated to generate augmentations. with probability 0.3 a word can be replaced by similar word. With probability of 0.4 i selected a sentence from training data to transform. This way i increased the relevant pairs data. 

#### Final results:
Training Metrics       :  'accuracy': 0.985 <br>
Validation/Test Metrics:  **'accuracy': 0.9325, 'precision': 0.8673 and 'recall': 0.8585** <br>
Note: split is 3200:800 here for training and validation/test

#### Next Steps/Improvements:
1. I think data augmentation is very important here, the augmentation i have used is based on wordnet(~ to the brute force approach i mentioned), we can create good augmentations using pretrained-bert(i tried but it takes time to generate) or even large models. 
2. I want to explain first what relevance means first quantitatively and also give proof for relevance. 
3. Work on preventing overfitting and handle class imbalance problem. 
4. we can use contrastive learning techniques. we have S1,S2 relevant to each other, we can find data in the same dataset that might not be relevant to S1 so we can have for example triplet loss term or info-NCE(with many -ve pairs) as extra term in final loss.

## Solution to Topic Labeling :
This follows last problem in many steps.
First Data:
```
We have 8 topics:
{'family': 64, 'work': 382, 'money': 99, 'medical': 99, 'emotional': 347, 'zp': 7, 'food': 1, 'miscellaneous': 1}
Very few examples are from zp, food, miscellaneous. Ideally this is also on class imbalance side if seen as pure multi-label classification problem.
```

The important thing here is input modeling: <br>
At training time, input is modelled as: input+' [SEP] '+ label <br>
At inference time, input is given just as a string

### Model
Input is passed through the Distilbert and we take the zero-position embedding([CLS]), let it be X(dim=768)
we project this 768-d X to 768-d again => A non-linear activation => Dropout => project to 8-d(logits)
Loss is CrossEntropyLoss

### Why this is working?
Since these are trained using masked langage modeling objective, the topic can be easily found with this input. In other interpretation, the model will be able quickly learn what comes after [SEP] based on the sentence tokens.. The model is able to learn in 3 epochs (few-shot learning ability of language models). With this modelling there is no need for data augmentation, focusing on overfitting..

#### Final results:
Training Metrics       :  'accuracy': 0.992 <br>
Validation/Test Metrics:  **'accuracy': 0.988** <br>
Note: split is 750:250 here for training and validation/test

#### Next steps/Improvements:
1. As mentioned in the notebook, some of these sentences can be multi-labelled i.e. from many topics (for example the 1, i got wrong can be [work, medicine]), instead of single label extend this to multi-label using language models.
2. The only cons of this model is inference time so next step is develop faster-inference models with same metrics. After seeing dataset, i also want to see what this gives: max_over_label(cosine-distance(x,label);x is word of sentence), this might actually give the good validation metric. Adding string-match too, this can be a very simple efficient model too!   
