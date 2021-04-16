from flask import Flask, render_template
import pandas as pd
import random
import networkx
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input,Activation,BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np


app = Flask(__name__)




def linpredmodel():
    model = Sequential()
    model.add(Input(5))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dense(2, activation='softmax'))
    
    model.load_weights('model.h5')
    
    return model
def common_neighbors(Graph, edges):
    """
    Input : graph as network class and edges
    Output: Common neigbors of the vertices connected by the given edge
    """
    result = []
    for edge in edges:
        src, dest = edge[0], edge[1]
        src_neighbor, dest_neighbor = set(Graph.neighbors(src)), set(Graph.neighbors(dest))
        count = len(src_neighbor.intersection(dest_neighbor))
        result.append((src, dest, count))
    return result

class Linkpredictor():
    def __init__(self,Graph,usersdata):

        # users list
        userslist={}
        for i in usersdata.index:
               userslist[usersdata.iloc[i]['id']]=usersdata.iloc[i]['name']
        self.users_list= userslist

        #graph
        self.graph=Graph  
        self.nodes=Graph.number_of_nodes() 
        self.graphfeatures = [common_neighbors,networkx.resource_allocation_index,networkx.jaccard_coefficient, networkx.adamic_adar_index, networkx.preferential_attachment]

        self.linkpredmodel=linpredmodel()



        
         
    def get_random_users(self,no):
        userdisplay={i:self.users_list[i]   for i in random.sample(range(0, self.nodes), no)}
        return userdisplay 
    
    def get_Edge_features(self,node1,node2):
        
        edgelist=[(str(node1),str(node2))]
        features=[]
        for func in self.graphfeatures:
            preds = func(self.graph, edgelist)
            f=[i[2] for i in preds]
            
            features.append(f[0])
        return np.array((features))
        
    def hops(self,node1,node2):
        short_path=networkx.shortest_path_length(self.graph,str(node1),str(node2))
        return short_path
    def __call__(self,node_id,count):
        friends=[]
        suggestions=[]
        for i in range(0,self.nodes):
          #print(i)  
            
          if(i==node_id):
              continue
          try:
               hops=self.hops(node_id,i)
               
               if(int(hops)==1):
                   friends.append([i,self.users_list[i]])
               else:
                   
                   if(hops<=3):
                       features=self.get_Edge_features(node_id,i)  
                       features=tf.expand_dims(features,0)
                       prediction=np.argmax(self.linkpredmodel.predict(features))
                       if(int(prediction)==1):
                           suggestions.append([i,self.users_list[i]])
                           
          except:
               pass                 
                           
        return friends[:count],suggestions[:count]  





@app.route('/user/<id>')
def user(id):
    print(int(id))
    friends,suggestion=fbnet(int(id),10)

    return render_template('userdetails.html', name=fbnet.users_list[int(id)],friends = friends,suggestion=suggestion)


@app.route('/')
def hello_world():


   userdisplay=fbnet.get_random_users(50)
   return render_template('users.html',users=userdisplay)

if __name__ == '__main__':

   data=pd.read_csv("./usernames.csv")
   path='facebook.txt'
   with open(path,'rb') as file:
       graph=networkx.read_edgelist(file)
   fbnet=Linkpredictor(graph,data)




   app.run()   


