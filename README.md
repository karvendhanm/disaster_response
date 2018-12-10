# disaster_response
Classify distress messages from people in disaster struck zones using NLP and supervised learning so that the messages could be forwarded to relevant agencies for relief.

![image](https://github.com/karvendhanm/disaster_response/blob/master/disaster_response.png)

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Code Execution Instruction](#samplecode)
6. [Licensing, Authors, and Acknowledgements](#licensing)
7. [Summary](#summary)

## Installation <a name="installation"></a>

The code should run with no issues using Python versions 3.*. Following are the libraries used in this project.

1. pandas
2. plotly
3. nltk
4. flask
5. sklearn
6. sqlalchemy


## Project Motivation<a name="motivation"></a>

For this project, I was interested in using figure-eight text [data](https://www.figure-eight.com/data-for-everyone/) from disaster struck zones for:

1. Classify the text and social media messages sent by people that require relief into thirty six different categories so that these messages can be forwarded to 
relevant aid agencies.


## File Descriptions <a name="files"></a>

* workspace/web_app/data: Folder contains .csv files(disaster_messages.csv and disaster_categories.csv) which are data files from figure-eight
dataset. Also, process_data.py which combines the aforementioned .csv files and saves the resulting dataframe in a sqlite database.
* workspace/web_app/models: Folder contains train_classifier.py which reads in from the created sqlite database, tokenizes the text messages,
builds a supervised model, and saves it as a pickle file.
* workspace/web_app/app: Folder contains run.py which initiates a flask instance as well as creates plotly data, and layout for couple of visualizations
for the webpage. Also, it has templates folder which has go.html, master.html which are used to create a webpage.


## Results<a name="results"></a>

1) I was able to categorize messages with around 90% accuracy with respect to all categories expect four categories where the accuracy is
around 80%. 

## Code Execution Instruction <a name="samplecode"></a>

1) To create and store the database: run the following command from the web_app folder 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
2) To create and the store the model: run the following command from the web_app folder 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
3) To use the app and classify text messages: Run the following command in the app's directory 'python run.py' and then go to 'http://0.0.0.0:3001/' where text messages can be classified in the webpage.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to figuure-eight for the [data](https://www.figure-eight.com/data-for-everyone/). Otherwise, feel free to use the code here as you would like!

## Summary<a name="summary"></a>

A dataset from figure-eight of text messages sent during disasters have been taken and the text messages have been classified into 36 categories. As disaster is the time when relief agencies are overwhelmed and can't sort through these messages manually or keyword search, these algorithms will help sort these message.

Overall was able to categorize messages with aound 90% accuracy with respect to all categories expect four categories where the accuracy is around 80%.


