import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import firebase_admin
import requests
from dotenv import load_dotenv
import os
import time
import io
from datetime import datetime
from firebase_admin import credentials, auth, initialize_app
from firebase_admin import firestore
from streamlit_option_menu import option_menu

# Set up the page configuration
st.set_page_config(page_title="Student Performance Prediction", layout="wide")

# Load the trained model
def load_model():
    with open(r"C:\Users\User\students.pkl", "rb") as file:
       stu_data = pickle.load(file)
    return stu_data

stu_data= load_model()

model = stu_data["logistic"]
scaler = stu_data["scaler"]
Parental_Involvement_le = stu_data["Parental_Involvement_le"]
Access_to_Resources_le = stu_data["Access_to_Resources_le"]
Extracurricular_Activities_le = stu_data["Extracurricular_Activities_le"]
Motivation_Level_le = stu_data["Motivation_Level_le"]
Internet_Access_le = stu_data["Internet_Access_le"]
Teacher_Quality_le = stu_data["Teacher_Quality_le"]
School_Type_le = stu_data["School_Type_le"]
Distance_from_Home_le = stu_data["Distance_from_Home_le"]
Parental_Education_Level_le = stu_data["Parental_Education_Level_le"]
Learning_Disabilities_le = stu_data["Learning_Disabilities_le"]
Peer_Influence_le = stu_data["Peer_Influence_le"]

# Firebase project details
load_dotenv()  # Loads from .env in current directory

FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")

if not FIREBASE_API_KEY:
    raise ValueError("FIREBASE_API_KEY is not loaded. Check your .env file.")

FIREBASE_AUTH_URL = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
# Initialize Firebase
if "firebase_initialized" not in st.session_state:
    if not firebase_admin._apps:  # Prevent multiple initializations
        cred = credentials.Certificate(r"C:\Users\User\OneDrive\Desktop\ML Projects\Student App\student-performance-342f5-e2ec5ea2756f.json")
        initialize_app(cred)
        st.session_state.firebase_initialized = True

# Initialize Firestore
db = firestore.client()

# Initialize session state variables
if "username" not in st.session_state:
    st.session_state.username = ""
if "useremail" not in st.session_state:
    st.session_state.useremail = ""
if "signedout" not in st.session_state:
    st.session_state.signedout = False
if "signout" not in st.session_state:
    st.session_state.signout = False

# Function for Login
def login():
    email = st.session_state.email
    password = st.session_state.password
    try:
        # Make a POST request to Firebase REST API for authentication
        response = requests.post(
            FIREBASE_AUTH_URL,
            json={"email": email, "password": password, "returnSecureToken": True}
        )
        if response.status_code == 200:
            data = response.json()
            #st.success("Login Successful")
            st.session_state.username = data.get("localId")
            st.session_state.useremail = data.get("email")
            st.session_state.signedout = True
            st.session_state.signout = True
        else:
            raise Exception(response.json().get("error", {}).get("message", "Login failed"))
    except Exception as e:
        st.error(f"Login failed: {str(e)}. Check email and password.")


# Function for Logout
def logout():
    # Clear session state variables
    st.session_state.username = ""
    st.session_state.useremail = ""
    st.session_state.signedout = False
    st.session_state.signout = False
    st.success("You have been logged out successfully!")
    
# Redirect to a specific page by simulating a rerun
def redirect_to_page(page):
    st.session_state.page = page
    

# Sidebar menu
if st.session_state.signedout:
    with st.sidebar:
        selected = option_menu(
            "Student Performance Prediction System",
            ["Home", "Prediction", "Visualizations", "Settings", "Logout"],
            icons=["house", "bar-chart-fill", "graph-up", "gear", "box-arrow-right"],
            menu_icon="book",
            default_index=0
        )
else:
    selected = "Account"

# Ensure session state for storing past predictions
if "past_predictions" not in st.session_state:
    st.session_state.past_predictions = []

# Function to enable downloading past performance insights
def download_performance_data(df):
    # Remove 'Timestamp' and 'UserEmail' columns before downloading
    df_filtered = df.drop(columns=['Timestamp', 'UserEmail'], errors='ignore')

    # Convert DataFrame to CSV format
    csv = df_filtered.to_csv(index=False)

    # Convert CSV to a byte stream
    csv_bytes = io.BytesIO(csv.encode())

    # Create a download button
    st.download_button(
        label="📥 Download Performance Data",
        data=csv_bytes,
        file_name="performance_data.csv",
        mime="text/csv"
    )

if selected == "Home":
    st.title("Welcome To The Home Page")

    if "useremail" not in st.session_state:
        st.warning("Please log in to view your predictions.")
        st.stop()

    try:
        predictions_ref = db.collection("predictions").where("UserEmail", "==", st.session_state.useremail).stream()
        past_predictions = [doc.to_dict() for doc in predictions_ref]

        if not past_predictions:
            st.warning("No past predictions available. Make a prediction first.")
        else:
            df_past = pd.DataFrame(past_predictions)
            df_past["Timestamp"] = pd.to_datetime(df_past["Timestamp"], errors="coerce")
            df_past = df_past.sort_values(by="Timestamp", ascending=False)

            if "Prediction" in df_past.columns:
                cols = [col for col in df_past.columns if col != "Prediction"] + ["Prediction"]
                df_past = df_past[cols]

            df_past_display = df_past.drop(columns=["Timestamp", "UserEmail"], errors="ignore")
            st.subheader("Past Predictions")
            st.dataframe(df_past_display)

            st.subheader("Performance Trends Over Time")
            prediction_map = {"Low Performance": 0, "Medium Performance": 1, "High Performance": 2}
            df_past["Prediction_numeric"] = df_past["Prediction"].map(prediction_map)
            st.line_chart(df_past.set_index("Timestamp")["Prediction_numeric"])

            st.subheader("AI-Based Recommendations")
            last_prediction = df_past.iloc[0]["Prediction"]
            if last_prediction == "Low Performance":
                st.error("⚠️ Student's performance is low. Consider increasing study hours and parental involvement.")
            elif last_prediction == "Medium Performance":
                st.warning("📈 Performance is average. Encouraging tutoring and structured study sessions may help.")
            else:
                st.success("🎉 High performance! Maintain the same habits and optimize learning strategies.")

            if 'download_performance_data' in globals():
                download_performance_data(df_past)

    except Exception as e:
        st.error(f"Error retrieving past predictions: {e}")

# Function for password reset
def reset_password():
    email = st.text_input("Enter your email to reset password", key="reset_email")
    if st.button("Send Reset Email"):
        try:
            reset_url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
            payload = {
                "requestType": "PASSWORD_RESET",
                "email": email
            }
            response = requests.post(reset_url, json=payload)
            
            if response.status_code == 200:
                st.success("A password reset email has been sent. Please check your inbox.")
            else:
                error_message = response.json().get("error", {}).get("message", "Failed to send reset email.")
                st.error(f"Error: {error_message}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Account Management Page
if selected == "Account":
    st.title("Account Management")

    if not st.session_state.signedout:
        # Add option buttons for Login and Signup
        action = st.radio("Choose an action", ["Login", "Sign Up","Forgot Password"], horizontal=True)

        if action == "Login":
            # Auto-fill saved credentials if "Remember Me" was used
            email = st.text_input("Email Address", key="email", value=st.session_state.get("saved_email", ""))
            password = st.text_input("Password", key="password", type="password", value=st.session_state.get("saved_password", ""))
            remember_me = st.checkbox("Remember Me", value=st.session_state.get("remember_me", False))
            if st.button("Login"):
                login()
                # Save credentials only if "Remember Me" is checked
                if remember_me:
                    st.session_state.saved_email = email
                    st.session_state.saved_password = password
                    st.session_state.remember_me = True
                else:
                    st.session_state.saved_email = ""
                    st.session_state.saved_password = ""
                    st.session_state.remember_me = False
                
            
        
        elif action == "Sign Up":
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            username = st.text_input("Enter Unique Username")
            if st.button("Create Account"):
                    try:
                        # Create a new user using Firebase Admin SDK
                        user = auth.create_user(email=email, password=password, uid=username)
                        
                        # Auto-login the user
                        st.session_state.username = user.uid
                        st.session_state.useremail = user.email
                        st.session_state.signedout = True
                        st.session_state.signout = True

                        st.success("Account Created Successfully! You are now logged in.")
                        time.sleep(2)
                        st.balloons()

                           # Clear input fields
                        st.session_state.signup_username = ""
                        st.session_state.signup_email = ""
                        st.session_state.signup_password = ""

                        # Force rerun to reflect login state
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        elif action == "Forgot Password":
            reset_password()

    else:
        st.text(f"Name: {st.session_state.username}")
        st.text(f"Email ID: {st.session_state.useremail}")
        st.button("Sign Out", on_click=logout)


# Initialize session state to store user inputs
if "x_new" not in st.session_state:
    st.session_state.x_new = {}

# Input options
Peer_Influence_options = ('Low','Negative','Neutral','Positive')
Parental_Involvement_options = ('High','Low','Medium')
Access_to_resources_options = ('High','Low','Medium')
Extracurricular_activities_options = ('No','Yes')
Motivation_level_options = ('High','Low','Medium')
Internet_Access_options = ('No','Yes')
Teacher_Quality_options = ('High','Low','Medium')
School_Type_options = ('Private','Public')
Distance_from_Home_options = ('Far','Moderate','Near')
Parental_education_level_options = ('College','High School','Postgraduate')
Learning_Disabilities_options = ('No','Yes')

if selected == "Prediction":
    st.title("Performance Prediction Using Machine Learning")

    # User input columns
    col1, col2, col3 = st.columns(3)

    # Collecting inputs
    with col1:
        Peer_Influence = st.selectbox("Peer Influence", Peer_Influence_options)
    with col2:
        Hours_Studied = st.slider("Hours Studied (hrs)", 0, 24, 1)
    with col3:
        Attendance = st.slider("Class Attendance (%)", 0, 100, 5)
    with col1:
        Previous_Scores = st.slider("Previous Scores (%)", 0, 100, 2)
    with col2:
        Tutoring_Sessions = st.slider("Tutoring Sessions", 0, 10, 2)
    with col3:
        Parental_Involvement = st.selectbox("Parental Involvement", Parental_Involvement_options)
    with col1:
        Access_to_resources = st.selectbox("Access to Resources", Access_to_resources_options)
    with col2:
        Distance_from_Home = st.selectbox("Distance from Home", Distance_from_Home_options)
    with col3:
        Parental_education_level = st.selectbox("Parental Education Level", Parental_education_level_options)
    with col1:
        Sleep_hrs = st.slider("Sleep Hours", 0, 8, 2)
    with col2:
        Motivation_level = st.selectbox("Motivation Level", Motivation_level_options)
    with col3:
        Teacher_Quality = st.selectbox("Teacher Quality", Teacher_Quality_options)
    with col1:
        School_Type = st.selectbox("School Type", School_Type_options)
    with col2:
        Learning_Disabilities = st.selectbox("Learning Disabilities", Learning_Disabilities_options)
    with col3:
        Physical_activity = st.slider("Physical Activity (hrs)", 0, 8, 0)
    with col1:
        Internet_Access = st.selectbox("Internet Access", Internet_Access_options)
    with col2:
        Extracurricular_activities = st.selectbox("Extracurricular Activities", Extracurricular_activities_options)

    # Predict Button
    if st.button("Predict"):
        try:
            # Construct input data
            x_input = [
                Peer_Influence,
                Hours_Studied,
                Attendance,
                Previous_Scores,
                Tutoring_Sessions,
                Parental_Involvement,
                Access_to_resources,
                Distance_from_Home,
                Parental_education_level,
                Sleep_hrs,
                Motivation_level,
                Teacher_Quality,
                School_Type,
                Learning_Disabilities,
                Physical_activity,
                Internet_Access,
                Extracurricular_activities
            ]

            x_new = np.array([x_input], dtype=object)

            # Encode categorical features (use index [0, col] since x_new is shape (1, 17))
            x_new[0, 0] = Peer_Influence_le.transform([x_new[0, 0]])[0]
            x_new[0, 5] = Parental_Involvement_le.transform([x_new[0, 5]])[0]
            x_new[0, 6] = Access_to_Resources_le.transform([x_new[0, 6]])[0]
            x_new[0, 7] = Distance_from_Home_le.transform([x_new[0, 7]])[0]
            x_new[0, 8] = Parental_Education_Level_le.transform([x_new[0, 8]])[0]
            x_new[0, 10] = Motivation_Level_le.transform([x_new[0, 10]])[0]
            x_new[0, 11] = Teacher_Quality_le.transform([x_new[0, 11]])[0]
            x_new[0, 12] = School_Type_le.transform([x_new[0, 12]])[0]
            x_new[0, 13] = Learning_Disabilities_le.transform([x_new[0, 13]])[0]
            x_new[0, 15] = Internet_Access_le.transform([x_new[0, 15]])[0]
            x_new[0, 16] = Extracurricular_Activities_le.transform([x_new[0, 16]])[0]

            # Convert to float and scale
            x_new = x_new.astype(float)
            x_scaled = scaler.transform(x_new)

            # Predict
            prediction = model.predict(x_scaled)

            # Interpret result
            result_text = (
                "High Performance" if prediction[0] == 2 else
                "Medium Performance" if prediction[0] == 1 else
                "Low Performance"
            )

            st.success(f"Predicted Student Performance: **{result_text}**")

            db.collection("predictions").add({
            "UserEmail": st.session_state.useremail,
            "Timestamp": firestore.SERVER_TIMESTAMP,
            "Prediction": result_text,
            "inputdata" : {
                "Peer Influence": Peer_Influence,
                "Hours Studied":Hours_Studied,
                "Attendance":Attendance,
                "Previous Scores":Previous_Scores,
                "Tutoring Sessions":Tutoring_Sessions,
                "Parental Involvement":Parental_Involvement,
                "Access to resources":Access_to_resources,
                "Distance from Home":Distance_from_Home,
                "Parental education level":Parental_education_level,
                "Sleep hrs":Sleep_hrs,
                "Motivation level":Motivation_level,
                "Teacher Quality":Teacher_Quality,
                "School Type":School_Type,
                "Learning Disabilities":Learning_Disabilities,
                "Physical activity":Physical_activity,
                "Internet Access":Internet_Access,
                "Extracurricular activities":Extracurricular_activities
            }

            # include other fields if needed
        })



            # Save inputs for visualization
            st.session_state.x_new = dict(zip(
                ["Peer_Influence", "Hours_Studied", "Attendance", "Previous_Scores", "Tutoring_Sessions",
                "Parental_Involvement", "Access_to_resources", "Distance_from_Home", "Parental_education_level",
                "Sleep_hrs", "Motivation_level", "Teacher_Quality", "School_Type", "Learning_Disabilities",
                "Physical_activity", "Internet_Access", "Extracurricular_activities"],
                x_input
            ))

        except Exception as e:
            st.error(f"Error during prediction: {e}")

            
           
# Visualization Page
if selected == "Visualizations":
    st.title("Visualizations based on User Inputs")

    if not st.session_state.x_new:
        st.warning("No input data available. Please provide inputs on the Home page.")
    else:
        # Convert session state data to DataFrame
        df = pd.DataFrame([st.session_state.x_new])

        # Bar Chart
        st.subheader("Bar Chart of User Inputs")
        st.bar_chart(df.T)

        # Table View
        st.subheader("User Input Data Table")
        st.dataframe(df)

# Function to reset password
def reset_password():
    email = st.text_input("Enter your email to reset password", key="reset_email")
    if st.button("Send Reset Email"):
        try:
            reset_url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
            payload = {
                "requestType": "PASSWORD_RESET",
                "email": email
            }
            response = requests.post(reset_url, json=payload)
            
            if response.status_code == 200:
                st.success("A password reset email has been sent. Please check your inbox.")
            else:
                error_message = response.json().get("error", {}).get("message", "Failed to send reset email.")
                st.error(f"Error: {error_message}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Function to delete user account
def delete_user():
    try:
        user = auth.get_user_by_email(st.session_state.useremail)
        auth.delete_user(user.uid)  # Delete from Firebase Authentication

        # Delete user data from Firestore
        user_ref = db.collection("users").document(st.session_state.username)
        user_ref.delete()

        # Clear session data
        st.session_state.username = ""
        st.session_state.useremail = ""
        st.session_state.signedout = False
        st.session_state.signout = False

        st.success("Your account has been deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting account: {e}")

# Initialize session state variables
if "username" not in st.session_state:
    st.session_state.username = ""
if "useremail" not in st.session_state:
    st.session_state.useremail = ""
if "signedout" not in st.session_state:
    st.session_state.signedout = False
if "signout" not in st.session_state:
    st.session_state.signout = False
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False  # To track confirmation state



# Settings Page
if selected == "Settings":
    st.title("Application Settings")
    

    if st.session_state.signedout:
        st.text(f"Name: {st.session_state.username}")
        st.text(f"Email ID: {st.session_state.useremail}")

        # Reset Password Section
        st.subheader("🔑 Reset Password")
        reset_password()

        # Delete Account Section
        st.subheader("⚠️ Delete Account")
        st.warning("This action is irreversible! Deleting your account will remove all your data.")
        if st.button("Delete My Account", help="This will permanently remove your account"):
            st.session_state.confirm_delete = True  # Show confirmation popup

        # Confirmation Popup
        if st.session_state.confirm_delete:
            with st.expander("⚠️ Are you sure you want to delete your account?"):
                st.error("Once deleted, your data **cannot be recovered**.")
                if st.button("Yes, Delete My Account"):
                    delete_user()
                if st.button("Cancel"):
                    st.session_state.confirm_delete = False  # Hide popup

    else:
        st.warning("Please log in to manage your account.")
        
        

    # Save Settings Button
    if st.button("Save Settings"):
        st.success("Settings Saved Successfully!")


# Logout logic
if selected == "Logout":
    logout()
    redirect_to_page("Account")





        
