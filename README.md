# Dissertation_Django_Server
The server uses NGROK to make the localhost available publicly. Make sure you have an account made. https://ngrok.com/

## Run Server

1. Change database settings in the settings.py folder (username/password)
2. You will need to add these tables to the database, run python manage.py makemigrations
3. Then, to apply to the database, run python manage.py migrate
4. run: python manage.py runserver 0.0.0.0:5000
5. run: ngrok http 5000
6. A base URL will be shown, which will need to be copied and added to the APICOnnections.java file in the Android app repository

## Files

- settings.py
All the server settings are here
- serealizer.py
Makes sure the data posted conforms to the schema
- views.py
All the logic for the queries are stored here
- models.py
The database schema for tables is stored here
- urls.py
Paths for the server which are appended to the NGROK base URL, and what method should run
- MLNew.py
Machine learning algorithms which use a global and personalised classifier. All code received from https://towardsdatascience.com/time-series-classification-for-fatigue-detection-in-runners-a-tutorial-d649e8eb322f
- Other Files
The CSV for testing the ML is included. "AllData.csv" is used for the training, whereas the other CSV files are used for testing with new data. There are 8 of each bus and car. 
