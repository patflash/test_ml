import streamlit as st
from multiapp import MultiApp
from pages import home,signIn,signUp # import your app modules here



app = MultiApp()



# Add all your application here
app.add_app("Home", home.app)
app.add_app("Login", signIn.app)
app.add_app("Sign Up", signUp.app)
    
# The main app
app.run()
