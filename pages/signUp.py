import streamlit as st
import sqlite3 



def app():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


    def add_userdata(username,password):
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
        conn.commit()

    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password",type='password')

    if st.button("Signup") :
        create_usertable()
        add_userdata(new_user, new_password)
        st.success("You have successfully created a valid Account")
        st.info("Go to Login Menu to login")

    
    

    

