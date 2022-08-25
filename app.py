
# import the required packages
from flask import Flask, render_template, request
import joblib
import pandas as pd

# instantiate the web-app
app = Flask(__name__)

# load our model pipeline object
model = joblib.load("reg2.joblib")

# outline the homepage or "default" page
# when a user visits this page, the home function will be run
@app.route("/")
def home():
    return render_template("index.html")

# outline the prediction page
# when a user visits the /predict page, the predict function will be run
@app.route('/predict', methods=['POST'])
def predict():
    
    # get input variables from form
    country = request.form.get('country')
    mainbranch = request.form.get('mainbranch')
    employmenttype = request.form.get('employmenttype')
    developertype = request.form.get('developertype')
    orgsize = request.form.get('orgsize')
    gender = request.form.get('gender')
    ethnicity = request.form.get('ethnicity')
    edlevel = request.form.get('edlevel')
    clevel = request.form.get('clevel')
    
    new_data = pd.DataFrame({"Country" : [country], "MainBranch" : [mainbranch], "Employment" : [employmenttype], "EdLevel" : [edlevel]
    , 'OrgSize' : [orgsize], 'Ethnicity' : [ethnicity], 'GenderEditted' : [gender], 'CareerLevel' : [clevel],  'DevType' : [developertype]})
    
    # apply model pipeline to the input data and extract probability prediction
    pred_ = model.predict(new_data)[0]
    # [1]
    # render the page using result.html and include the predicted probability
    return render_template("result.html", prediction_text = f"{pred_}")
    
if __name__ == "__main__":
    app.run(debug=True)