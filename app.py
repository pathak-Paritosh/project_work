import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Function to load all models from a .pkl file
def load_models(filename):
    return joblib.load(filename)

# Load all models
all_models = load_models("all_models.pkl")

# Function to make predictions using a specific model
def predict_drug(model_name, new_data):
    model = all_models[model_name]
    predicted_drug = model.predict(new_data)
    return predicted_drug

def main():
    # Load the trained model
    # LRclassifier = load_model()

    # Title of the app
    st.title("Drug Prediction App")

    # Sidebar with input fields
    with st.sidebar:
        st.subheader("Enter Patient Information")
        age = st.slider("Age", min_value=0, max_value=100, value=30)
        sex = st.radio("Sex", ["Male", "Female"])
        bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
        cholesterol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
        na_to_k = st.number_input("Na_to_K")

    # Preprocess the input data
    new_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "BP": [bp],
        "Cholesterol": [cholesterol],
        "Na_to_K": [na_to_k]
    })

    bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
    category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']

    bin_NatoK = [0, 9, 19, 29, 50]
    category_NatoK = ['<10', '10-20', '20-30', '>30']

    # Define all possible categories for each categorical variable
    categories = {
        'Sex': ['F', 'M'],
        'BP': ['HIGH', 'LOW', 'NORMAL'],
        'Cholesterol': ['HIGH', 'NORMAL']
    }

    # Create a DataFrame with the new data point and all possible categories
    new_data_df = pd.DataFrame(new_data)

    new_data_df['Age_binned'] = pd.cut(new_data_df['Age'], bins=bin_age, labels=category_age)
    new_data_df = new_data_df.drop(['Age'], axis = 1)

    new_data_df['Na_to_K_binned'] = pd.cut(new_data_df['Na_to_K'], bins=bin_NatoK,labels=category_NatoK)
    new_data_df = new_data_df.drop(['Na_to_K'], axis = 1)


    for col, possible_values in categories.items():
        new_data_df[col] = pd.Categorical(new_data_df[col], categories=possible_values)

    # Perform one-hot encoding
    new_data_encoded = pd.get_dummies(new_data_df)

    # Make predictions
    predicted_drug_by_LR = predict_drug("LRclassifier", new_data_encoded)
    predicted_drug_by_KN = predict_drug("KNclassifier", new_data_encoded)
    predicted_drug_by_SVC = predict_drug("SVCclassifier", new_data_encoded)
    predicted_drug_by_CNB = predict_drug("NBclassifier1", new_data_encoded)
    predicted_drug_by_GNB = predict_drug("NBclassifier2", new_data_encoded)
    predicted_drug_by_DT = predict_drug("DTclassifier", new_data_encoded)
    predicted_drug_by_RF = predict_drug("DTclassifier", new_data_encoded)

    # Display the predicted drug
    # st.subheader("Prediction")
    # st.write("Predicted Drug By Logistic Regression:", predicted_drug_by_LR[0])
    # st.write("Predicted Drug By KNN:", predicted_drug_by_KN[0])
    # st.write("Predicted Drug By SVM:", predicted_drug_by_SVC[0])
    # st.write("Predicted Drug By Categorial NB:", predicted_drug_by_CNB[0])
    # st.write("Predicted Drug By Gaussian NB:", predicted_drug_by_GNB[0])
    # st.write("Predicted Drug By DT:", predicted_drug_by_DT[0])
    # st.write("Predicted Drug By RF:", predicted_drug_by_RF[0])

    # ----------------------------------------------------------------------------
    # Create a DataFrame to hold the predictions
    predictions_df = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN", "SVM", "Categorical NB", "Gaussian NB", "Decision Tree", "Random Forest"],
        "Predicted Drug": [predicted_drug_by_LR[0], predicted_drug_by_KN[0], predicted_drug_by_SVC[0], 
                        predicted_drug_by_CNB[0], predicted_drug_by_GNB[0], predicted_drug_by_DT[0], 
                        predicted_drug_by_RF[0]]
    })

    # Display the predictions in a table
    st.subheader("Predictions")
    st.table(predictions_df)

    # ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
