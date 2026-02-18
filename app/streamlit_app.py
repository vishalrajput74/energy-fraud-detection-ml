import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Streamlit App

def main():
    st.title("AI Model App")
    st.write("This app allows you to upload data, preprocess it, visualize it, train a model, and make predictions.")

    # File Upload Section
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data uploaded successfully!")
            st.dataframe(data.head())

            # Sidebar options
            st.sidebar.title("Options")
            action = st.sidebar.selectbox("Choose an action", [
                "None", "Preprocess Data", "Visualize Data", "Train Model", "Make Prediction"
            ])

            if action == "Preprocess Data":
                # Preprocess Data
                st.subheader("Data Preprocessing")
                if st.checkbox("Remove missing values"):
                    data.dropna(inplace=True)
                    st.success("Missing values removed.")
                    st.dataframe(data.head())

            elif action == "Visualize Data":
                # Visualization Section
                st.subheader("Data Visualization")
                st.write("Visualize pairwise relationships in your data.")
                try:
                    sns.pairplot(data)
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error during visualization: {e}")

            elif action == "Train Model":
                # Model Training Section
                st.subheader("Train a Random Forest Model")
                if st.button("Train Model"):
                    try:
                        # Splitting features and target (assuming the last column is the target)
                        X = data.iloc[:, :-1]
                        y = data.iloc[:, -1]

                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Train the model
                        model = RandomForestClassifier(random_state=42)
                        model.fit(X_train, y_train)

                        # Test the model
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)

                        st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}")

                        # Display classification report
                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred))

                        # Save the model in session state for prediction
                        st.session_state.model = model
                    except Exception as e:
                        st.error(f"Error during model training: {e}")

            elif action == "Make Prediction":
                # Prediction Section
                st.subheader("Make Predictions")
                if "model" in st.session_state:
                    model = st.session_state.model
                    input_values = st.text_input("Enter feature values separated by commas", "")

                    if st.button("Predict"):
                        try:
                            input_data = [float(val) for val in input_values.split(",")]

                            # Check if input matches model's feature requirements
                            if len(input_data) != model.n_features_in_:
                                st.error(f"Expected {model.n_features_in_} features, but got {len(input_data)}")
                            else:
                                prediction = model.predict([input_data])
                                st.success(f"Predicted class: {prediction[0]}")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                else:
                    st.error("No trained model found. Please train the model first.")
        except Exception as e:
            st.error(f"Error reading the file: {e}")

# Run the app
if __name__ == "__main__":
    main()