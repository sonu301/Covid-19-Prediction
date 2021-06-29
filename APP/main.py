from flask import Flask,render_template,request
import pickle

 

file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()

app = Flask(__name__)



@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        mydict=request.form
        fever=int(mydict['fever'])
        age=int(mydict['age'])
        pain=int(mydict['pain'])
        runnyNose=int(mydict['runnynose'])
        diffBreath=int(mydict['diffBreath'])
        infprob=clf.predict_proba([[fever,pain,age,runnyNose,diffBreath]])[0][1]
        
        return render_template('show.html',inf=round(infprob*100) )
    
    # return f"Covid-probability is {inf}"
    return render_template('index.html')





if __name__=='__main__':
    app.run(debug=True)


