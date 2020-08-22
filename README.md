# disaster-response
This project is build to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages; by a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.
The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

To run 

# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
