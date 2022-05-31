import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import sqlite3 



def app():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


    def login_user(username,password):
        c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
        data = c.fetchall()
        return data

    def view_all_users():
        c.execute('SELECT * FROM userstable')
        data = c.fetchall()
        return data


    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password",type='password')
    if st.sidebar.checkbox("Login"):
        #connexion 
            create_usertable()
            result = login_user(username,password)
            if result :
                st.success(f'Logged In as {username}')

                task = st.selectbox("Menu",["Setting","Add User","Profile"])
                if task == "Setting":
                    #Centre de Verification, dévaluation et d'amélioration du ML
                    st.subheader("Performance machine")
                    ddos_dns = pd.read_csv('datasets/DrDoS_DNS.csv')
                    ddos_ldap = pd.read_csv('datasets/DrDoS_LDAP.csv')
                    ddos_mssql = pd.read_csv('datasets/DrDoS_MSSQL.csv')
                    ddos_ntp = pd.read_csv('datasets/DrDoS_NTP.csv')
                    ddos_udp = pd.read_csv('datasets/DrDoS_UDP.csv')

                    datas=pd.concat([ddos_dns,ddos_ldap,ddos_mssql,ddos_ntp,ddos_udp],axis=0)
                    st.write(datas.head())
                    st.write(f'Taille du dataset : {datas.shape}')

                    X = datas.drop([' Label'],axis=1)
                    y = datas[' Label']

                    st.subheader("Pré-Traitement des données")
                    #encodage de données
                    encoder = LabelEncoder()
                    z = encoder.fit_transform(y.values)

                    if st.checkbox("Affichier le nuage de points"):
                        fig = plt.figure(figsize=(10 , 5))
                        plt.scatter(X[' Flow Duration'],X['Total Length of Fwd Packets'],c=z,alpha=0.8)
                        st.pyplot(fig)

                    label_number = []

                    #Selection de variable
                    for col in X.select_dtypes('number'):
                        label_number.append(col)
                        
                    X = X[label_number]

                    st.subheader("Evaluation des Performances")

                    from sklearn.model_selection import train_test_split

                    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
                    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

                    X_test = X_test.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)

                    if st.checkbox("Afficher le données du Train_set et Test_set "):
                        st.write(f"Train set : {X_train.shape} , \n Test set {X_test.shape}")

                    from sklearn.ensemble import RandomForestClassifier

                    model = RandomForestClassifier()

                    model.fit(X_train,y_train)
                    score = model.score(X_test,y_test)
                    if st.checkbox("Afficher le score accurency"):
                        st.write(f"Votre score accurrency est de : {score}")
                    
                    st.subheader("Validation des données")
                    #Cross Validation 
                    from sklearn.model_selection import cross_val_score

                    val_score = cross_val_score(RandomForestClassifier(), X_train, y_train, cv=5, scoring='accuracy')
                    if st.checkbox("Afficher les données de validation"):
                        if st.checkbox("Tableau de Validation"):
                            st.write('Nombre de Split = 5')
                            st.write(val_score)
                        if st.checkbox("Afficher la courbe de validation"):
                            scr = []
                            for k in range(1,50):
                                score = cross_val_score(RandomForestClassifier(n_estimators=k), X_train, y_train, cv=5).mean()
                                scr.append(score)
                            
                            fig = plt.figure(figsize=(10 , 5))
                            plt.plot(scr)
                            st.pyplot(fig)

                    st.subheader("Meilleur CV: meilleurs paramètres")
                    #Grid Search CV
                    from sklearn.model_selection import GridSearchCV
                    param_grid = {'n_estimators': np.arange(1,100)}

                    grid = GridSearchCV(RandomForestClassifier(),param_grid,cv=5)

                    grid.fit(X_train,y_train)

                    model = grid.best_estimator_
                    nb_estimator = grid.best_params_

                    if st.checkbox("Meilleurs Score et Meilleurs paramètres obtenu :"):
                        st.write(f"best score : {grid.best_score_}")
                        st.write(f"nombres estimateurs : {nb_estimator['n_estimators']}")

                    #Matrice de Confusion
                    st.write("Matrice de Confusion")
                    from sklearn.metrics import confusion_matrix
                    if st.checkbox("Matrice de Confusion"):
                        st.write(confusion_matrix(y_test,model.predict(X_test)))

                    #Courbe d'apprentissage
                    st.write("Courbe d'apprentissage")
                    from sklearn.model_selection import learning_curve
                    if st.checkbox("Courbe d'apprentissage"):
                        N, train_score, val_score = learning_curve(model,X_train,y_train, train_sizes=np.linspace(0.3,1.0,10),cv=5)
                    
                        fig = plt.figure(figsize=(10 , 5))
                        plt.plot(N, train_score.mean(axis=1),label='train')
                        plt.plot(N, val_score.mean(axis=1), label='validation')
                        plt.xlabel('train_sizes')
                        plt.legend()
                        st.pyplot(fig) 
                    

                
                elif task == "Add User" :
                    st.subheader("Ajouter un nouvel utilisateur")
                    
                elif task == "Profile" :
                    st.subheader("User Profile")
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result ,columns=["Username","Password"])
                    st.dataframe(clean_db)
            else :
                st.warning("Incorrect Username / Password")
    else:
        st.info("Veuillez vous connecter pour avoir accès")

    

    

