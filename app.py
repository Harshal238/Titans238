from os import read
from flask import Flask, render_template,request
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set()
data = pd.read_csv("static/musicdataset.csv")

df = data.drop(columns=['id', 'name', 'artists', 'release_date', 'year'])
df.corr()

from sklearn.preprocessing import MinMaxScaler
datatypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
normarization = data.select_dtypes(include=datatypes)
for col in normarization.columns:
    MinMaxScaler(col)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
features = kmeans.fit_predict(normarization)
data['features'] = features
MinMaxScaler(data['features'])



app = Flask(__name__)
@app.route('/')
def main():
  return render_template('index.html')



@app.route("/recommend",methods=['POST'])
def home():
  text=request.form['firstname']
  # data=pd.read_csv("static/musicdataset.csv")
  
  recommendations.recommend(text)
  # recommend(data,text,10)
  return render_template("table.html","index.html")

        
    
class Spotify_Recommendation():
    def __init__(self, dataset):
        self.dataset = dataset
    def recommend(self, songs, amount=5):
        distance = []
        song = self.dataset[(self.dataset.name.str.lower() == songs.lower())].head(1).values[0]
        rec = self.dataset[self.dataset.name.str.lower() != songs.lower()]
        for songs in tqdm(rec.values):
            d = 0
            for col in np.arange(len(rec.columns)):
                if not col in [1, 6, 12, 14, 18]:
                    d = d + np.absolute(float(song[col]) - float(songs[col]))
            distance.append(d)
        rec['distance'] = distance
        rec = rec.sort_values('distance')
        columns = ['artists', 'name']
        var=(rec[columns][:amount])
        df=pd.DataFrame(var)
        html=df.to_html()
        text_file=open("templates/table.html","w")
        text_file.write(html)
        text_file.close()
recommendations = Spotify_Recommendation(data)
   
if __name__ == "__main__":
    app.run(debug=True)