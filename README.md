# Disaster Response Pipeline Project


# Installations
- Python version 3.8.5
- All packages were installed with the Anaconda distribution
- Packages used:
	- pandas
	- pickle
	- nltk
	- sklearn
	- sqlalchemy
	- sys
	- re

# Project Motivation
This goal of this project is to read in messages data, apply text processing, and classify the categories of the messages for the Udacity Data Science Nanodegree program's second project.  The dataset was provided by FigureEight.


# File descriptions

| Name| Description |
| ----------- | ----------- |
| process_data.py|  Script to load, clean, and save data to a sqlite database. |
| train_classifier.py| Script to load the data from teh sqlite database, perform text processing, and apply, evaluate, and save a classification model to the dataset. |
| run.py | Script to run the flask application. |
| disaster_categories.csv | FigureEight data containing ids and message categories|
| disaster_messages.csv   | FigureEight data containing ids and messages|
| DisasterResponse.db| Sqlite database containing processed data from FigureEight|
| model.pkl| Serialized classification model|
| go.html| Standard template for web application|
| master.html| Standard template for web application|

# How to interact with this project
The .py files within the repository were designed for others to replicate the analysis if desired.    


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Data preparation summary.  
- Messages were joined with the categories data to obtain associated categories for each disaster message 
- Categories were cleaned and converted to binary columns


## Results summary
- Model performance:

# Licensing, Authors, Acknowledgements
Thank you to FigureEight for making the data accessible.  Thank you to Udacity for providing templates of the folder and file structure needed for this project.