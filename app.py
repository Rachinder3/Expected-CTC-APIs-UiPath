from flask import Flask, jsonify, request
from TransformationPckg.Transformations import *
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os


pipeline = joblib.load("pipeline_important.joblib")
columns_uipath = ['Education', 'PHD_Specialization', 'Current_CTC', 'Inhand_Offer', 'Last_Appraisal_Rating']


app = Flask(__name__)

@app.route("/transform_via_postman", methods=['GET','POST'])
def transform_via_postman():
    if request.method=='POST':

        ##############  Reading the data
        education = request.json["Education"]
        phd_spec = request.json["PHD_Specialization"]
        current_ctc = int(request.json["Current_CTC"])
        inhand_offer = request.json["Inhand_Offer"]
        last_appraisal_ratings = request.json["Last_Appraisal_Rating"]


        ############### Buidling the feature array
        feature_array = [education,phd_spec,current_ctc,inhand_offer,last_appraisal_ratings]

        ############### doing the transformation
        df = pd.DataFrame(np.array(feature_array).reshape(1,-1), columns=columns_uipath)

        data_transformed = pipeline.transform(df)
        data_transformed_df = pd.DataFrame(data_transformed)

        return (data_transformed_df.to_json())



@app.route("/transform_via_uipath", methods=['GET','POST'])
def transform_via_uipath():
    if request.method=='POST':
        #data = request.values.get("degree")

        ##############  Reading the data
        education = request.values.get("Education")
        phd_spec = request.values.get("PHD_Specialization")
        current_ctc = int(request.values.get("Current_CTC"))
        inhand_offer = request.values.get("Inhand_Offer")
        last_appraisal_ratings = request.values.get("Last_Appraisal_Rating")

        ############### Buidling the feature array
        feature_array = [education, phd_spec, current_ctc, inhand_offer, last_appraisal_ratings]

        ############### doing the transformation
        df = pd.DataFrame(np.array(feature_array).reshape(1, -1), columns=columns_uipath)

        data_transformed = pipeline.transform(df)
        data_transformed_df = pd.DataFrame(data_transformed)

        return (data_transformed_df.to_json())

@app.route("/bulk_transform_via_uipath", methods = ['POST','GET'])
def bulk_transform_via_uipath():
    if request.method == 'POST':

        # reading the json input
        
        input_string = request.values.get("input")
        
        ######## reading the data
        bulk_upload_df = pd.read_json(input_string)


        ######### transforming the data
        bulk_data_transformed = pipeline.transform(bulk_upload_df)
        bulk_data_transformed_df = pd.DataFrame(bulk_data_transformed)

        ## returning the data
        return (bulk_data_transformed_df.to_json())

@app.route("/", methods = ['POST','GET'])
def index():
    return "<h1> Expected CTC API </h1>"




if __name__ == '__main__':
    port = os.environ.get("PORT",5000)
    app.run(debug=False, host="0.0.0.0",port=port)







