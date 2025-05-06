import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Load the saved models
working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Custom CSS for modern styling
st.markdown("""
    <style>
        /* General Styling */
        body { font-family: Arial, sans-serif; background-color: #f4f6f9; color: #333; }
        
        /* Header Styling */
        .main-header {
            text-align: center;
            padding: 40px;
            background-color: #f0f8ff;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 40px;
        }
        .main-header h1 { color: #0046a8; margin-bottom: 10px; }
        .main-header p { font-size: 1.2em; color: #333; }
            
            .main-header:hover {
             transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }

        /* Health Tips Section */
        .health-tips {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
        }
        .tip-box {
            flex: 1 1 30%;
            padding: 20px;
            background-color: #e0f7fa;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .tip-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        }
        .tip-box h3 { color: #0046a8; margin-bottom: 10px; }
        .tip-box p { color: #333; font-size: 1em; }

        /* Testimonials Section */
        .testimonials {
            background-color: #fff3e0;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin: 40px auto;
            max-width: 1200px;
        }
        .testimonials h2 { text-align: center; color: #ff9800; margin-bottom: 30px; }
        .testimonial {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            margin-bottom: 20px;
        }
        .testimonial p {
            font-style: italic;
            color: #555;
            max-width: 600px;
        }
        .testimonial .name {
            margin-top: 10px;
            font-weight: bold;
            color: #ff5722;
        }

        /* Button Styling */
        .stButton button {
            background-color: #0046a8;
            color: #fff;
            border-radius: 8px;
            transition: 0.3s;
            font-size: 1em;
            padding: 10px 20px;
            margin-top: 10px;
        }
        .stButton button:hover {
            background-color: #003580;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        }

        /* Result Styling */
        .result {
            font-size: 1.5em;
            font-weight: bold;
            color: #1b5e20;
            text-align: center;
            padding: 10px;
            background-color: #e8f5e9;
            border-radius: 8px;
            margin-top: 20px;
        }
        .recovery {
            margin-top: 20px;
            padding: 15px;
            border-left: 4px solid #0046a8;
            background-color: #f0f4f9;
            border-radius: 5px;
            color: #333;
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .health-tips, .testimonials {
                flex-direction: column;
                align-items: center;
            }
            .tip-box, .testimonial {
                width: 100%;
            }
        }
        .dis-header {
     background: linear-gradient(135deg, #43a7f8 0%, #dc7ced 100%);
    color: #fff;
    padding: 20px;
    text-align: center;
    font-size: 1.5em;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    transition: transform 0.3s, box-shadow 0.3s;
}

.dis-header:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
}

    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    selected = option_menu(
        'Health Assistant',
        ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction", 'Disease Distribution'],
        icons=['house', 'activity', 'heart', 'activity', 'pie-chart'],
        menu_icon='hospital-fill',
        default_index=0
    )

# Home Page
if selected == 'Home':
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>Welcome to Health Assistant</h1>
        <p>Your reliable tool for early health predictions and personalized insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Health Tips Section
    st.markdown("""
    <div class="health-tips">
        <div class="tip-box">
            <h3>Diabetes Prediction</h3>
            <p>Assess your risk for diabetes and receive recommendations to manage it effectively.</p>
        </div>
        <div class="tip-box" style="background-color: #ffebee;">
            <h3 style="color: #b71c1c;">Heart Disease Prediction</h3>
            <p>Understand your heart health and take preventive steps for a healthier life.</p>
        </div>
        <div class="tip-box" style="background-color: #f3e5f5;">
            <h3 style="color: #6a1b9a;">Parkinson’s Prediction</h3>
            <p>Evaluate your Parkinson's risk factors and gain early insights.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Testimonials Section
    st.markdown("""
    <div class="testimonials">
        <h2>What Our Users Say</h2>
        <div class="testimonial">
            <p>"Health Assistant has been a game-changer for me. The diabetes prediction tool helped me take proactive steps towards a healthier lifestyle."</p>
            <div class="name">- Sarah M.</div>
        </div>
        <div class="testimonial">
            <p>"I was skeptical at first, but the heart disease prediction provided accurate insights that motivated me to improve my heart health."</p>
            <div class="name">- John K.</div>
        </div>
        <div class="testimonial">
            <p>"The Parkinson’s prediction feature gave me peace of mind by assessing my risk factors early on."</p>
            <div class="name">- Emily R.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Optional: Add a Call to Action Button
    st.markdown("""
    <div style="text-align: center; margin-top: 40px;">
        <a href="#quick_assessment">
            <button style="background-color: #ff9800; color: white; padding: 15px 30px; border: none; border-radius: 8px; 
                           font-size: 1.2em; cursor: pointer;">
                Start Your Health Assessment
            </button>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Diabetes Prediction Page

elif selected == 'Diabetes Prediction':
    st.markdown("""
    <div class="dis-header">
        <h1>Diabetes Prediction using Machine Learning</h1>
        
    </div>
    """, unsafe_allow_html=True)
    st.subheader('')
    st.markdown("Please fill in the details below:")

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.expander("Diabetes Input", expanded=True):
                Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
                Glucose = st.number_input('Glucose Level', min_value=0.0, step=0.1)
                BloodPressure = st.number_input('Blood Pressure', min_value=0.0, step=0.1)

        with col2:
            with st.expander("Diabetes Input", expanded=True):
                SkinThickness = st.number_input('Skin Thickness', min_value=0.0, step=0.1)
                Insulin = st.number_input('Insulin Level', min_value=0.0, step=0.1)
                BMI = st.number_input('BMI', min_value=0.0, step=0.1)

        with col3:
            with st.expander("Diabetes Input", expanded=True):
                DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01)
                Age = st.number_input('Age', min_value=0, step=1)

    if st.button('Predict Diabetes'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        if all(x >= 0 for x in user_input):
            diab_prediction = diabetes_model.predict([user_input])
            result = 'The Person is Diabetic' if diab_prediction[0] == 1 else 'Not Diabetic'
            st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)

            if result == 'The Person is Diabetic':
                st.markdown("<div class='recovery'><strong>Recovery and Management Tips</strong></div>", unsafe_allow_html=True)
                st.markdown("""
                    - **Maintain a balanced diet**: Include fiber, whole grains, and lean proteins.
                    - **Exercise regularly**: Aim for moderate exercise most days.
                    - **Monitor blood sugar levels**.
                    - **Stay hydrated**.
                    - **Get adequate sleep**: 7-8 hours.
                """)
        else:
            st.error("Please enter valid input values.", icon="🚨")

# Heart Disease Prediction Page
elif selected == 'Heart Disease Prediction':
    st.markdown("""
    <div class="dis-header">
        <h1>Heart Disease Prediction using Machine Learning</h1>
        
    </div>
    """, unsafe_allow_html=True)
    st.subheader('')
    st.subheader('')
    st.markdown("Please fill in the details below:")

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.expander("Heart Disease Input", expanded=True):
                age = st.number_input('Age', min_value=0, step=1)
                sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')

        with col2:
            with st.expander("Heart Disease Input", expanded=True):
                cp = st.selectbox('Chest Pain Type', options=list(range(0, 4)))
                trestbps = st.number_input('Resting Blood Pressure', min_value=0, step=1)

        with col3:
            with st.expander("Heart Disease Input", expanded=True):
                chol = st.number_input('Serum Cholestoral (mg/dl)', min_value=0, step=1)
                fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: 'False' if x == 0 else 'True')

        with col1:
            with st.expander("Heart Disease Input", expanded=True):
                restecg = st.selectbox('Resting Electrocardiographic results', options=[0, 1, 2])
                thalach = st.number_input('Max Heart Rate Achieved', min_value=0, step=1)

        with col2:
            with st.expander("Heart Disease Input", expanded=True):
                exang = st.selectbox('Exercise Induced Angina', options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
                oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, step=0.1)

        with col3:
            with st.expander("Heart Disease Input", expanded=True):
                slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2])
                ca = st.number_input('Major Vessels Colored by Fluoroscopy', min_value=0, step=1)
                thal = st.selectbox('Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', options=[0, 1, 2])

    if st.button('Predict Heart Disease'):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        if all(x >= 0 for x in user_input):
            heart_prediction = heart_disease_model.predict([user_input])
            result = 'Has Heart Disease' if heart_prediction[0] == 1 else 'Does Not Have Heart Disease'
            st.markdown(f"<div class='result'>{result}</div>", unsafe_allow_html=True)

            if result == 'Has Heart Disease':
                st.markdown("<div class='recovery'><strong>Recovery and Management Tips</strong></div>", unsafe_allow_html=True)
                st.markdown("""
                    - **Eat heart-healthy foods**: Include vegetables, fruits, whole grains, and lean proteins.
                    - **Stay physically active**.
                    - **Monitor cholesterol and blood pressure**.
                    - **Quit smoking and avoid excessive alcohol**.
                    - **Manage stress**.
                """)
        else:
            st.error("Please enter valid input values.", icon="🚨")

# Parkinson's Prediction Page
elif selected == "Parkinson's Prediction":
    st.markdown("""
    <div class="dis-header">
        <h1>Parkinson’s Disease Prediction using Machine Learning</h1>
        
    </div>
    """, unsafe_allow_html=True)
    st.subheader('')
    st.markdown('<div class="header"><h1></h1></div>', unsafe_allow_html=True)
    st.markdown("Please enter the following details to predict the likelihood of Parkinson's Disease:")

    # Arrange inputs in columns for a clean, modern layout
    col1, col2, col3, col4, col5 = st.columns(5)

    # Collect input data with placeholder for modern look
    with col1:
        fo = st.text_input('MDVP: Fo (Hz)', placeholder='e.g., 119.992')

    with col2:
        fhi = st.text_input('MDVP: Fhi (Hz)', placeholder='e.g., 157.302')

    with col3:
        flo = st.text_input('MDVP: Flo (Hz)', placeholder='e.g., 74.997')

    with col4:
        Jitter_percent = st.text_input('MDVP: Jitter (%)', placeholder='e.g., 0.005')

    with col5:
        Jitter_Abs = st.text_input('MDVP: Jitter (Abs)', placeholder='e.g., 0.00005')

    with col1:
        RAP = st.text_input('MDVP: RAP', placeholder='e.g., 0.003')

    with col2:
        PPQ = st.text_input('MDVP: PPQ', placeholder='e.g., 0.005')

    with col3:
        DDP = st.text_input('Jitter: DDP', placeholder='e.g., 0.009')

    with col4:
        Shimmer = st.text_input('MDVP: Shimmer', placeholder='e.g., 0.02')

    with col5:
        Shimmer_dB = st.text_input('MDVP: Shimmer (dB)', placeholder='e.g., 0.17')

    with col1:
        APQ3 = st.text_input('Shimmer: APQ3', placeholder='e.g., 0.01')

    with col2:
        APQ5 = st.text_input('Shimmer: APQ5', placeholder='e.g., 0.02')

    with col3:
        APQ = st.text_input('MDVP: APQ', placeholder='e.g., 0.03')

    with col4:
        DDA = st.text_input('Shimmer: DDA', placeholder='e.g., 0.04')

    with col5:
        NHR = st.text_input('NHR', placeholder='e.g., 0.005')

    with col1:
        HNR = st.text_input('HNR', placeholder='e.g., 21.04')

    with col2:
        RPDE = st.text_input('RPDE', placeholder='e.g., 0.43')

    with col3:
        DFA = st.text_input('DFA', placeholder='e.g., 0.66')

    with col4:
        spread1 = st.text_input('Spread1', placeholder='e.g., -4.31')

    with col5:
        spread2 = st.text_input('Spread2', placeholder='e.g., 0.34')

    with col1:
        D2 = st.text_input('D2', placeholder='e.g., 2.45')

    with col2:
        PPE = st.text_input('PPE', placeholder='e.g., 0.13')

    # Prediction button and result display
    parkinsons_diagnosis = ''
    if st.button("Predict Parkinson's"):
        try:
            # Convert inputs to float
            user_input = [
                float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs),
                float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB),
                float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR),
                float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2),
                float(D2), float(PPE)
            ]

            # Make prediction
            parkinsons_prediction = parkinsons_model.predict([user_input])

            # Result interpretation and display with modern look
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease."
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease."

            st.markdown(f"<div class='result'>{parkinsons_diagnosis}</div>", unsafe_allow_html=True)

            # Recovery and management tips
            if parkinsons_prediction[0] == 1:
                st.markdown("<div class='recovery'><strong>Recovery and Management Tips</strong></div>", unsafe_allow_html=True)
                st.markdown("""
                    - **Regular check-ups**: Consult with a healthcare provider regularly.
                    - **Physical activity**: Engage in light exercise, like walking or stretching.
                    - **Medication adherence**: Take medications as prescribed.
                    - **Balanced diet**: Include fresh fruits, vegetables, and lean proteins.
                    - **Support system**: Maintain a strong support network of family and friends.
                """)
        except ValueError:
            st.error("Please ensure all fields are filled in correctly with numerical values.", icon="🚨")

# Sample data (use your actual DataFrame)
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8, 4, 10, 10, 1, 5, 7, 0, 7, 1, 1, 3, 8, 7, 9],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125, 110, 168, 139, 189, 166, 100, 118, 107, 103, 115, 126, 99, 196, 119],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96, 92, 74, 80, 60, 72, 0, 84, 74, 30, 70, 88, 84, 90, 80],
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0, 0, 0, 0, 23, 19, 0, 47, 0, 38, 30, 41, 0, 0, 35],
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0, 0, 0, 0, 846, 175, 0, 230, 0, 83, 96, 235, 0, 0, 0],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0, 37.6, 38.0, 27.1, 30.1, 25.8, 30.0, 45.8, 29.6, 43.3, 34.6, 39.3, 35.4, 39.8, 29.0],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232, 0.191, 0.537, 1.441, 0.398, 0.587, 0.484, 0.551, 0.254, 0.183, 0.529, 0.704, 0.388, 0.451, 0.263],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54, 30, 34, 57, 59, 51, 32, 31, 31, 33, 32, 27, 50, 41, 29],
    'Outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Binning functions
def bin_diabetes_pedigree(value):
    if value < 0.5:
        return 'Low'
    elif 0.5 <= value < 1.0:
        return 'Medium'
    else:
        return 'High'

def bin_glucose(value):
    if value < 100:
        return 'Low'
    elif 100 <= value < 140:
        return 'Medium'
    else:
        return 'High'

def bin_age(value):
    if value < 30:
        return 'Young'
    elif 30 <= value < 50:
        return 'Middle-aged'
    else:
        return 'Older'

def bin_blood_pressure(value):
    if value < 60:
        return 'Low'
    elif 60 <= value < 80:
        return 'Normal'
    else:
        return 'High'

# Apply bins to create new columns for each category
df['DiabetesPedigreeCategory'] = df['DiabetesPedigreeFunction'].apply(bin_diabetes_pedigree)
df['GlucoseCategory'] = df['Glucose'].apply(bin_glucose)
df['AgeCategory'] = df['Age'].apply(bin_age)
df['BloodPressureCategory'] = df['BloodPressure'].apply(bin_blood_pressure)

# Function to create pie chart for a specific category with a fixed small size
def create_pie_chart(data, title, colors, size=(2, 2)):
    fig, ax = plt.subplots(figsize=size) 
    counts = data.value_counts()
    ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2},
        textprops={'fontsize': 8} 
    )
    ax.set_title(title, fontweight="bold", fontsize=10, color="#444444")
    return fig

# Colors for each pie chart
colors = ['#A0C4FF', '#FFB5A7', '#FF99AC']

# Page selection
selected = st.sidebar.selectbox("Select a Page", ["Home", "Disease Distribution", "Heart Disease Prediction", "Parkinson's Prediction"])

if selected == 'Disease Distribution':
    st.markdown("""
    <div class="dis-header">
        <h1>Disease Distribution Analysis by Categories</h1>
        
    </div>
    """, unsafe_allow_html=True)
    st.subheader('') 
    st.subheader("")
    
    # Display each pie chart in the Disease Distribution section only
    st.pyplot(create_pie_chart(df['DiabetesPedigreeCategory'], "Diabetes Pedigree Function Distribution", colors, size=(2, 2)))
    st.pyplot(create_pie_chart(df['GlucoseCategory'], "Glucose Level Distribution", colors, size=(2, 2)))
    st.pyplot(create_pie_chart(df['AgeCategory'], "Age Distribution", colors, size=(2, 2)))
    st.pyplot(create_pie_chart(df['BloodPressureCategory'], "Blood Pressure Distribution", colors, size=(2, 2)))