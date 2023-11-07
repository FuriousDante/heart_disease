import streamlit as st 
import pickle

model = pickle.load(open('model.pkl', 'rb'))

def predict_pro(list1):
    
    print("inside predict function")
    val=model.predict(list1)

    if val[0] == 1:
        prediction = "yes"
    else:
        prediction = "no"
    
    
    return prediction

def main():

    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    bpscaler = MinMaxScaler()
    cholscaler = MinMaxScaler()
    thalachcaler = MinMaxScaler()
    oldpeakscaler = MinMaxScaler()

    data= pd.read_csv("data-problem-statement-1-heart-disease.csv")

    bpscaler.fit(data["trestbps"].to_numpy().reshape(-1,1))
    cholscaler.fit(data["chol"].to_numpy().reshape(-1,1))
    thalachcaler.fit(data["thalach"].to_numpy().reshape(-1,1))
    oldpeakscaler.fit(data["oldpeak"].to_numpy().reshape(-1,1))

    pred_arr = []
    gender_dict = {"Male":1,"Female":0}
    yes_no_dict = {"Yes":1,"No":0}

    
    st.title("Heart Disease Classification Tool")
    html_temp = """
    <div style="background-color:#b3db86;padding:8px">
    <h2 style="color:black;text-align:center;">Predicting Heart Diease</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    tenure = st.slider("Enter your age", min_value=0, max_value=100)
    pred_arr.extend([tenure])
    # print(pred_arr)

    gender = st.selectbox("Please select your gender",("Male", "Female"))
    pred_arr.extend([gender_dict[gender]])
    # print(pred_arr)

    cp = st.slider("Select your chest Pain level", min_value=0, max_value=3)
    pred_arr.extend([cp])
    # print(pred_arr)
    
    trestbps = st.slider("Enter your Resting Blood Pressure", min_value=0, max_value=200)
    # pred_arr.extend([trestbps])
    v1 = bpscaler.transform([[trestbps]])
    pred_arr.extend(v1[0])

    chol= st.slider("Enter your Cholestrol level", min_value=0, max_value=1000)
    # pred_arr.extend([chol])
    v2 = cholscaler.transform([[chol]])
    pred_arr.extend(v2[0])

    fbs = st.selectbox("Is your Fasting Bllod Sugar > 120?", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[fbs]])
    # print(pred_arr)

    restecg = st.slider("Select your Restecg level", min_value=0, max_value=2)
    pred_arr.extend([restecg])
    # print(pred_arr)

    thalach = st.slider("Enter your Maximum Heart rate achieved", min_value=0, max_value=500)
    # pred_arr.extend([thalach])
    v3 = thalachcaler.transform([[thalach]])
    pred_arr.extend(v3[0])
    # print(pred_arr)

    exang = st.selectbox("Do you have Exercise induced angina",("Yes", "No"))
    pred_arr.extend([yes_no_dict[exang]])
    # print(pred_arr)

    oldpeak = st.slider("Enter your old peak", min_value=0, max_value=500)
    # pred_arr.extend([oldpeak])
    v4 = oldpeakscaler.transform([[oldpeak]])
    pred_arr.extend(v4[0])
    # print(pred_arr)

    slope = st.slider("Enter ythe level of slope", min_value=0, max_value=2)
    pred_arr.extend([slope])
    # print(pred_arr)

    ca = st.slider("Enter the number of major vessesls colored by flurosopy", min_value=0, max_value=3)
    pred_arr.extend([ca])
    # print(pred_arr)

    thal = st.slider("Enter your thal", min_value=0, max_value=2)
    pred_arr.extend([thal])
    
    result=""
    if st.button("Predict"):
        
        print(len(pred_arr))
        # pred_arr = [[1,89.35,89.35,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0]]
        result = predict_pro([pred_arr])

        print("Printing result")
        print(result)
        
        if result=="yes":
                st.error('The patient is suffering from a heart disease!')
        else:
             st.success("the patient is fine!", icon="✅")
    if st.button("About"):
        st.text("Best model used is GaussianNaivebayes, with approx 89% accuracy on test data")

main()
