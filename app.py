import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__,template_folder='template')
model = pickle.load(open('hccluster.pkl','rb'))
@app.route('/')
def home():
  
    return render_template("hcindex.html")
@app.route('/predict',methods=['GET'])
def predict():
  '''
  For rendering results on HTML GUI
  '''
  income1 = int(request.args.get('income1'))
  score1 = int(request.args.get('score1'))
  income2 = int(request.args.get('income2'))
  score2 = int(request.args.get('score2')) 
  income3 = int(request.args.get('income3'))
  score3 = int(request.args.get('score3')) 
  income4 = int(request.args.get('income4'))
  score4 = int(request.args.get('score4')) 
  income5 = int(request.args.get('income5'))
  score5 = int(request.args.get('score5'))
  predict = model.fit_predict([[income1,score1 ],[income2,score2], [income3,score3],[income4,score4], [income5,score5]])
  return render_template('hcindex.html', prediction_text='Model  has predicted  : {}'.format(predict))
if __name__ =='__main__':
 app.run()