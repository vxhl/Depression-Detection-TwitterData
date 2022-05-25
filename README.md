# Project Showcase : Twitter Depression Detection 🙂🕵️‍♀️
Implementing advanced NLP 💬 techniques using torchtext for detecting depression 🙎‍♂️ from user tweets

My notion file if you wish to read through my research and learnings through this project. 
https://notch-curiosity-62d.notion.site/Detecting-depression-in-Social-Media-using-Twitter-Data-b8c98f0d7a5b4fe1992cfb9af160517e

I have taken an already collected and combined dataset from https://github.com/swcwang/depression-detection since they were kind enough to open source a dataset for depression detection that they built and combined to get optimal results with their models

you can look over their Data Collection model [here]{"https://github.com/swcwang/depression-detection#collecting-data"}

Here I have used Torchtext, a Pytorch library for preprocessing of the twitter data.

Most of the preprocessing has been inspired by code from this article : https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8 This is my first time implementing NLP with Pytorch!

### Detecting and Removing abnormalies in our tweets
Now we remove the unnecessary stuff from our tweets like URLs ( That may point towards advertisement ), Mentions ( That are most likely an anomaly case ), Emojis (Since they are random ), Hashtag symbols and Numbers. We also work on removing all symbols and punctuations expect for ., ! and ?

Next we tokenize our tweets using spacy by making tweet_clean(s) as an nlp object.

### Building our Train-Val-Test Datasets
I was honestly pretty confused on the significant differences between a test and validation datasets,

So I referred to this insanely dedicated article to have a better understanding for them. https://machinelearningmastery.com/difference-test-validation-datasets/#:~:text=%E2%80%93%20Validation%20set%3A%20A%20set%20of,of%20a%20fully%2Dspecified%20classifier.&text=These%20are%20the%20recommended%20definitions%20and%20usages%20of%20the%20terms.

Suppose that we would like to estimate the test error associated with fitting a particular statistical learning method on a set of observations. The validation set approach […] is a very simple strategy for this task. It involves randomly dividing the available set of observations into two parts, a training set and a validation set or hold-out set. The model is fit on the training set, and the fitted model is used to predict the responses for the observations in the validation set. The resulting validation set error rate — typically assessed using MSE in the case of a quantitative response—provides an estimate of the test error rate.

### Loading our Vector embeddings and build_vocab
Torchtext makes loading of pretrained word vectors very easy. Just mention the name of the pretrained word vector (e.g. glove.6B.50d, fasttext.en.300d, etc.) and torchtext will download that particular vector and then you can use it in embedding layer.

### Loading Data in Batches
We will use the BucketIterator to access the Dataloader. It sorts data according to length of text, and groups similar length text in a batch, thus reducing the amount of padding required. It pads the batch according to the max length in that particular batch

### Models and Training
For model training I followed the example in https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8 along with some more studies from the https://github.com/swcwang/depression-detection who made small modifications by adding dropouts to lessen overfitting.

Our mode uses a pretrained embedding layer from glove that we imported using torchtext, a bidirecetional GRU and a concat pooling method where we perform average pool and max pool and then concatenate the results.

We define our vocab_size our vector embedding dimensions and hidden layers for our neural network.

### Final Results
```
test_loss, test_acc = evaluate(m, iter(test_batch_it), loss_fn, len(test_batch_it))

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
Test Loss: 0.466 | Test Acc: 77.66%
```

### Improving Accuracy of our Model
Now that we have gained an average accuracy of 76% let us look into how to improve this performance.

### Hyper Parameter Tuning
- On reducing the batch size we gain an average val_acc of 78%
- Reducing the number of hidden layers -> Decreases the accuracy so we reset
- Increasing the dropout rate also reduces the accuracy so we skip
- Applying Data Augmentation

I was pretty lost after this so didn't go forward with improving the model, if anyone comes across this and is skilled enough I would be happy to get some help!
