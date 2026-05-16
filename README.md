## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has four major parts :
1. model.py - This contains code fot our Machine Learning model to predict employee salaries absed on trainign data in 'diabetes3.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
3. request.py - This uses requests module to call APIs already defined in app.py and dispalys the returned value.
4. templates - This folder contains the HTML template to allow user to enter patient details and displays the predicted status of the patient(diabetic or not).

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000



Enter valid numerical values in all 8 input boxes and hit Predict.

If everything goes well, you should  be able to see the predicted patient status on the HTML page!
![image](https://user-images.githubusercontent.com/10856626/73582991-7f51b380-4498-11ea-89e3-7557ed661241.png)
![image](https://user-images.githubusercontent.com/10856626/73582997-85e02b00-4498-11ea-97c5-34433fb71ac8.png)



```
