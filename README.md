# EECS595 Final Project (18FA) Team19

This is a sentiment analysis on Yelp review(https://github.com/tyrozty/Sentimental-Analysis#eecs595)

All of the project files are uploaded on Google Drive, to run the code, go to: https://drive.google.com/open?id=1FxfjoHy3-_2VM4LzcVVZzZs-OLly_CPZ

### Reading list:

[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

[Standford cs225n Attention](http://web.stanford.edu/class/cs224n/lectures/lecture12.pdf)

[Text Classification](https://github.com/TobiasLee/Text-Classification)

* [Dataset]():The dataset is from the Yelp Dataset Challenge, the original link is: https://www.yelp.com/dataset/download. To run the program from very beginning, I highly recommend that you download the data and unzip it, and only put the 'yelp_dataset_review.json' file into '[/yelp_dataset](file)'

* [Installation](): python3, tensorflow, word2vec, NLTK, WorldCloud.
To build the running environment, run:
```commandline
$ pip3 install -r requirments.txt
```
* [Training](): There are three steps in the training part. The first step is data preprocessing. Second, the word embedding will be implemented. Finally, the data will be fed into our model and trained. So if there is no data in the [/yelp_dataset](file), the data_loader will be run to generate the training data. Also, if there are no embedding models in [/model/embedding/](file), the data_loader will be run to train the word embedding model. If everything is prepared in your system, then run the following command to train your model.
```commandline
$ python3 main_train.py
```
There is a word2vec emebdding training process which will take about 5 hours to load the data and train the word2vec model. If you want to skip this step, just download the trained model through Google Drive (the link can be found at the top).
The model should be downloaded from the [/mdoel/embedding/](file) on Google Drive.
If you want to skip the data generation step and train the Attentive CNN model directly, just download the processed data from [/yelp_dataset/data_split/](file) on Google Drive. 

* [File Description]():
```
NLP\
    model\
        embedding\
           POStag.pkl -- POStag model for word embedding 
            word2vec.model -- word2vec mdoel trianed based on the Yelp reviews
    yelp_dataset\
        label2text.pkl -- a dict key is the stars and the value is the reveiews at this star
        negative_text.pkl -- the review texts labeled by negative sentiment
        neutral_text.pkl -- the review texts labeled by neutral sentiment
        positive_text.pkl -- the review texts labeled by positve sentiment
        yelp_academic_dataset_reviews.csv -- raw data saved as csv file**
        yelp_academic_dataset_reviews.json -- raw data saved as json file
    ACNNModel.py -- the Attentive Convolutional Neural Network model based on Tensorflow
    baseline_emb.py -- the models, including linear model and decision tree model for comparing with performance of ACNN model        
    data_analysis.py -- visualizing the text level data by using WorldCloud
    data_loader.py -- laoding raw data, training word2vec and postag model for each word in this step
    main_train.py -- the main function that contians all the logic of our data loading, model training and testing
    output_DecisionTree.txt -- the result of the decision tree model
    output_lr.txt -- the result of the linear model
    preprocess.py -- the data preprocessing file
    res_plot.py -- the file that plots the training and testing results in training process
    ohter pkl files -- the results that saved for plotting and further research

```
