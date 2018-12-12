# EECS595

This is a sentiment analysis on Yelp review

### Reading list:

[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)

[Standford cs225n Attention](http://web.stanford.edu/class/cs224n/lectures/lecture12.pdf)

[Text Classification](https://github.com/TobiasLee/Text-Classification)

* [Dataset]():The dataset are YelpData, the original link is: https://www.yelp.com/dataset/challenge, to run the program from very begining, I highly recommend you download the data, and unzip it, only put the 'yelp_dataset_review.json' into '[/yelp_dataset](file)'

* [Installation](): python3, tensorflow, word2vec, NLTK, WorldCloud.To build the runing environment, run:
```commandline
$ pip3 install -r requirments.txt
```
* [Training](): there are three steps in training part, the first step is data preprocessing, then, the word embedding will be implemented, in the last step those data will be feed in to our model and trained. So if there are no data in the [/yelp_dataset](file), the data_loader will be run to generate the training data. Also, if there are no embedding models in [/model](file), the data_loader will be run to train the word emebdding file. If everything is prepared in your system, then run following command to train your model.
```commandline
$ python3 main_train.py
```
There is word2vec emebdding training progress which will takes a long time to load the data and training the word2vec model. If you want to skip this step, just download the trained model through Google Dirve(the link is attached at bottom), the model should be download from the [/mdoel/embedding/](file) in Google Drive.
If you want to skip data generating and train the Attentive CNN model directly, just download the processed data from [/yelp_dataset/data_split/](file) in Google Drive. 
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
    ACNNModel.py -- the Attentive Convolutional Neural Network model based on tensorflow
    baseline_emb.py -- the mdoels, including linear model and decision tree mdoel for comparing with performance of ACNN model        
    data_analysis.py -- visualizing the text level data by using WorldCloud
    data_loader.py -- laoding raw data, training word2vec and postag model for each word in this step
    main_train.py -- the main function that contians all the logic of our data loading, model training and testing
    output_DecisionTree.txt -- the result of the decision tree model
    output_lr.txt -- the reslut of the linear modek
    preprocess.py -- the data preprocessing file
    res_plot.py -- the file that plots the training and testing resluts in training progress
    ohter pkl files -- the resluts that saved for plotting and further research

```
The whole file was uploaded on Google Drive, to run the code, go to: https://drive.google.com/open?id=1FxfjoHy3-_2VM4LzcVVZzZs-OLly_CPZ