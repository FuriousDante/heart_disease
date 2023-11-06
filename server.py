import streamlit as st 
import pickle

print('Successfully executed ')

model = pickle.load(open('model.pkl', 'rb'))

def predict_pro(list1):
    
    print("inside predict function")
    val=model.predict(list1)

    if val[0] == 1:
        prediction = "yes"
    else:
        prediction = "no"
    
    
    return prediction

# def processing(l):
#     # Your dictionary
#     category_to_binary = {'Male': 1, 
#                            'Fiber Optic': (0, 1), 'No': (0, 0)}

# # Extract the tuple for 'DSL'
# dsl_tuple = category_to_binary['DSL']

# print(dsl_tuple)
# print("Inside Processing")
# print(l)

def main():

    pred_arr = []
    gender_dict = {"Male":1,"Female":0}
    yes_no_dict = {"Yes":1,"No":0}


    lines_dict = {"Yes":(1,0), "No":(0,1), "NA":(0,0)}
    internet_service_dict = {"DSL":(1,0), "Fibre Optic":(0,1), "No":(0,0)}
    online_security_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)} 
    online_backup_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)}
    device_protection_dict = {"Yes":(1,0), "No":(1,0), "NA":(1,0)}
    tech_support_dict = {"Yes":(1,0), "No":(1,0), "NA":(1,0)}
    stream_tv_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)}
    stream_movies_dict = {"Yes":(0,1), "No":(1,0), "NA":(1,0)}
    contract_dict = {"Monthly":(0,0), "One":(1,0), "Two":(0,1)}
    mail_dict = {"Bank":(0,0,0), "Credit":(1,0,0), "Electronic":(0,1,0), "Mailed": (0,0,1)}
    

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
    
    trestbps = st.slider("Enter your trestbps", min_value=0, max_value=200)
    pred_arr.extend([trestbps])

    chol= st.slider("Enter your chol", min_value=0, max_value=1000)
    pred_arr.extend([chol])

    fbs = st.selectbox("Select fbs", ("Yes", "No"))
    pred_arr.extend([yes_no_dict[fbs]])
    # print(pred_arr)

    restecg = st.slider("Select your Restecg level", min_value=0, max_value=2)
    pred_arr.extend([restecg])
    # print(pred_arr)

    thalach = st.slider("Select your thalash level", min_value=0, max_value=500)
    pred_arr.extend([thalach])
    # print(pred_arr)

    exang = st.selectbox("Do you have exang",("Yes", "No"))
    pred_arr.extend([yes_no_dict[exang]])
    # print(pred_arr)

    oldpeak = st.slider("Enter your old peak", min_value=0, max_value=500)
    pred_arr.extend([oldpeak])
    # print(pred_arr)

    slope = st.slider("Enter your slope", min_value=0, max_value=2)
    pred_arr.extend([slope])
    # print(pred_arr)

    ca = st.slider("Enter your ca", min_value=0, max_value=3)
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
        
        st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Built with Streamlit, By Mayur Parab and Siddharth Jadhwani")

main()
