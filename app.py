from flask import Flask, render_template, request, flash
import pandas as pd
import json
import plotly 
import plotly.express as px 
import plotly.io as pio


app = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"

@app.route("/hello")
def index():
	flash("what's your name?")
	return render_template("index.html")

@app.route("/greet", methods=['POST', 'GET'])
def greeter():
	flash("Hi " + str(request.form['name_input']) + ", great to see you!")
	return render_template("index.html")

#Dataset = pd.read_csv('penalty_signals.csv', encoding="UTF-8")
#Signals = pd.read_csv('signals16_21.csv', encoding="UTF-8")

#@app.route("/data", methods=['POST', 'GET'])
#def post_data():
#    Data_jsonStr = Signals.to_json()
#    pythonObj = json.loads(Data_jsonStr)
#    #response = pythonObj
#    fig = px.line(Signals, x='datetime', y='APEF')
#    ploting_plotly = pio.show(fig)
#    return ploting_plotly

#@app.route("/signals")
#def signals():
#    return render_template('historical.html')
 

#app.run()
