# Diabetes-Detection-Model

This is a Streamlit web application that predicts the likelihood of diabetes in a patient based on various health parameters. The prediction is made using a Random Forest Classifier trained on the Pima Indians Diabetes Database.

## Features

- **User Input Interface**: Allows users to input patient data through sliders for various health metrics.
- **Diabetes Prediction**: Uses a machine learning model to predict whether the patient is diabetic or not.
- **Data Visualization**: Visualizes user input data against the training dataset to provide context and comparison.
- **Model Accuracy Display**: Shows the accuracy of the machine learning model.

## How It Works

### User Input

Users input their health data through an intuitive sidebar interface. The following parameters are collected:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- Blood Pressure: Diastolic blood pressure (mm Hg)
- Skin Thickness: Triceps skinfold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- Diabetes Pedigree Function: A function which scores likelihood of diabetes based on family history
- Age: Age of the patient

### Data Processing

- The input data is collected and formatted into a dataframe that matches the structure of the training data.
- This dataframe is then fed into the trained Random Forest model to make a prediction.

### Machine Learning Model

- **Model Used**: Random Forest Classifier
- **Training Data**: The Pima Indians Diabetes Database
- **Model Training**: The model is trained using the training portion of the dataset and tested for accuracy using the test portion.

### Visualization

- **Scatter Plots**: The application generates scatter plots for various health parameters (like Age vs. Glucose, Age vs. BMI) comparing the user's input data with the training data. The user's data point is highlighted in these plots to provide a clear visual comparison.

### Output

- **Prediction**: The application displays whether the user is predicted to be diabetic or not.
- **Model Accuracy**: The accuracy of the model is displayed to provide users with an understanding of how reliable the predictions are.


## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Use the sliders in the sidebar to input patient data.

3. View the prediction and various visualizations comparing the patient's data with the training dataset.

## File Structure

- `app.py`: Main application script.
- `diabetes.csv`: Dataset used for training the model.


## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- pillow
- plotly

