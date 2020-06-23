import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    from PIL import Image
    image = Image.open('bg.jpg')
    st.image(image)
    st.title("ML Web App using Streamlit")
    st.sidebar.title("Binaray classification app")
    st.markdown("Mushroom Classification")
    st.sidebar.markdown("Mushroom ?")

    # this stores the data instead of streamlit loading all time
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv('MUSHROOM.CSV')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        return x_train, x_test, y_train, y_test

    # plotting model metrics
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confision Metrics")
            plot_confusion_matrix(model, x_test, y_test,
                                  display_labels=class_names)
            st.pyplot()

        if 'Roc Curve' in metrics_list:
            st.subheader("Roc Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonus']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("SVM", "Logistic Regression", "Random Forest"))

    if classifier == 'SVM':
        st.sidebar.subheader("Model Hyperparameters")
        # defining parameters
        C = st.sidebar.number_input("C", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma", ("scale", "auto"), key='gamma')
        # Metrics to select
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ('Confusion Matrix', 'Roc Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("SVM Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall :", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        # defining parameters
        C = st.sidebar.number_input("C", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider(
            "Max no of Iter", 100, 1000, key='max_iter')
        #kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key='kernel')
        #gamma = st.sidebar.radio("Gamma", ("scale", "auto"), key='gamma')
        # Metrics to select
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ('Confusion Matrix', 'Roc Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall :", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        # defining parameters
        max_depth = st.sidebar.number_input(
            "max_depth", 1, 100, step=1, key='max_depth')
        n_estimators = st.sidebar.number_input(
            "n_estimators", 100, 1000, step=1, key='n_estimator')
        max_iter = st.sidebar.slider(
            "Max no of Iter", 100, 1000, key='max_iter')
        #kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key='kernel')
        #gamma = st.sidebar.radio("Gamma", ("scale", "auto"), key='gamma')
        # Metrics to select
        metrics = st.sidebar.multiselect(
            "What metrics to plot?", ('Confusion Matrix', 'Roc Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='Classify'):
            st.subheader("Logistic Regression Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("precision: ", precision_score(
                y_test, y_pred, labels=class_names).round(2))
            st.write("Recall :", recall_score(
                y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    # Display data on the webapp

    if st.sidebar.checkbox("Show Raw Data", False):  # default is false
        st.subheader("Displaying the data set")
        st.write(df.head())

    # split the data


if __name__ == '__main__':
    main()
