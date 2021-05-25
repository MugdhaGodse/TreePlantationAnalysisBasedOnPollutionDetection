from flask import Flask,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)


@app.route("/get_analysis",methods=["GET","POST"])
def func():

    if request.method=="POST":
        inm=request.form.get("industryname")
        ar = int(request.form.get("area"))
        aa = int(request.form.get("availarea"))
        co2= int(request.form.get("CO2"))
        no2= int(request.form.get("NO2"))
        so2= int(request.form.get("SO2"))
        #print(ar,co2,no2,so2)
        with open('model_decisiontree','rb')as f:
            model=pickle.load(f)
        with open('model_regression','rb')as f:
            reg=pickle.load(f)
        trees = model.predict([[co2, no2, so2]])
        number_of_trees = reg.predict([[ar]])
        d={"Type of Industries":inm,"CO2":co2,"SO2":so2,"NO2":no2,"Area in sq. m":ar,"area for planting":aa}

        dd=pd.DataFrame(d)
        dd.to_csv(r"dataset2.csv",header=False)
    else:
        tem="Invalid Data"
        b="Invaid Data"

    return render_template('mainpage.html',plant=trees,No=number_of_trees)

app.run(debug=True)