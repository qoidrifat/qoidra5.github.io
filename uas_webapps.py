import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

st.set_page_config(page_title='UAS PENDATA')
st.markdown("<h1 style='text-align: center;'>UAS PENAMBANGAN DATA</h1>", unsafe_allow_html=True)
st.markdown("<p>Nama : Qoid Rif'at</p>", unsafe_allow_html=True)
st.markdown("<p>NIM : 210411100160</p>", unsafe_allow_html=True)
st.markdown("<p>Kelas : Penambangan Data B</p>", unsafe_allow_html=True)
st.write("---")



description, preprocessing, modelling, implementation = st.tabs(["DATASET", "PREPROCESSING", "CLASSIFICATION", "IMPLEMENTATION"])


with description:
    st.write("#### Dataset yang digunakan mengenai mental stress yang dapat dilihat pada tabel dibawah ini:")
    df = pd.read_csv('data.csv')
    st.dataframe(df)
    st.write("###### Sumber Dataset : https://www.kaggle.com/datasets/chtalhaanwar/mental-stress-ppg")
    st.write(" Dataset ini berisi informasi tentang mental stress dari tiap individu yang diteliti. Terdapat dua jenis labels yaitu normal dan stress. ")
    
with preprocessing:
    st.markdown("<h1 style='text-align: center;'>NORMALISASI DATA</h1>", unsafe_allow_html=True)
    st.markdown("<h3>RUMUS</h3>", unsafe_allow_html=True)
    img = Image.open('normalisasi.jpg')
    st.image(img, use_column_width=False, width=250)
    st.write("<h3>Dataset Sebelum di Normalisasi</h3>", unsafe_allow_html=True)
    # Mendefinisikan Varible X dan Y
    X = df.drop(columns=['labels'])
    y = df['labels'].values
    df
    st.subheader("Pemisahan Kolom labels Sebagai Atribut Target")
    X
    df_min = X.min()
    df_max = X.max()

    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    # features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.labels).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1': [dumies[0]],
        '2': [dumies[1]]
    })

    st.write(labels)

with modelling:
    # NORMALISASI NILAI X
    scaler = MinMaxScaler()
    # scaler.fit(features)
    # scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    # features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)
    # Nilai X training dan Nilai X testing
    training, test = train_test_split(
        scaled_features, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(
        y, test_size=0.2, random_state=1)  # Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        mlp_model = st.checkbox('ANNBackpropagation')

        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set labelss
        y_pred = gaussian.predict(test)

        y_compare = np.vstack((test_label, y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        # Gaussian Naive Bayes
        # gaussian = GaussianNB()
        # gaussian = gaussian.fit(training, training_label)

        # probas = gaussian.predict_proba(test)
        # probas = probas[:,1]
        # probas = probas.round()

        # gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        # KNN
        K = 10
        knn = KNeighborsClassifier(n_neighbors=K)
        knn.fit(training, training_label)
        knn_predict = knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label, knn_predict))

        # Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        # Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label, dt_pred))
        # Menggunakan 2 layer tersembunyi dengan 100 neuron masing-masing
        mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
        mlp.fit(training, training_label)
        mlp_predict = mlp.predict(test)
        mlp_accuracy = round(100 * accuracy_score(test_label, mlp_predict))

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(
                    gaussian_akurasi))
            if k_nn:
                st.write(
                    "Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree:
                st.write(
                    "Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
            if mlp_model:
                st.write(
                    'Model ANN (MLP) accuracy score: {0:0.2f}'.format(mlp_accuracy))
    

with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        gender = st.selectbox('Masukkan subject ID (sub_1-sub_27):', ('0', '1'))
        hemoglobin = st.number_input('Masukkan Labels (Noemal/Stress): ')
        mch = st.number_input('Masukkan Nilai 0 (0-999): ')
        mchc = st.number_input('Masukkan Nilai 1 (0-999): ')
        mcv = st.number_input(
            'Masukkan MCV : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi dibawah ini:',
                             ('Naive Bayes', 'K-NN', 'Decision Tree', 'ANNBackpropaganation'))

        apply_pca = st.checkbox("Include PCA")

        prediksi = st.form_submit_button("Submit")

        if prediksi:
            inputs = np.array([
                hemoglobin,
                mch,
                mchc,
                mcv,
                int(gender) 
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            # if apply_pca:
            #     pca = PCA(n_components=2)
            #     X_pca = pca.fit_transform(X)
            #     input_norm = pca.fit_transform(input_norm)

            if apply_pca and X.shape[1] > 1 and X.shape[0] > 1:
                pca = PCA(n_components=min(X.shape[1], X.shape[0]))
                X_pca = pca.fit_transform(X)
                input_norm = pca.transform(input_norm)

            if model == 'Naive Bayes':
                mod = gaussian
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'KNN Classifier':
                mod = knn
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'Decision Tree Classifier':
                mod = dt
                if apply_pca:
                    input_norm = pca.transform(input_norm)
            if model == 'ANN Classifier':
                mod = mlp
                if apply_pca:
                    input_norm = pca.transform(input_norm)

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
            ada = 1
            tidak_ada = 0
            if input_pred == ada:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'Pasien di Diagnosis penyakit ANEMIA')
            else:
                st.write('Berdasarkan hasil Prediksi Menggunakan Permodelan ',
                         model, 'Pasien Tidak di Diagnosis penyakit ANEMIA')
