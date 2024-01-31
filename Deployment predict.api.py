# Import library yang diperlukan
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat dataset
dataset = pd.read_csv('data.csv')

# Preprocessing dataset
# Misalnya, jika terdapat data yang perlu diubah formatnya atau dihapus kolom tertentu
dataset = dataset.dropna()  # Menghapus baris dengan nilai yang hilang
dataset['feature1'] = dataset['feature1'].astype(int)  # Mengubah tipe data kolom feature1 menjadi integer

# Memisahkan dataset menjadi data training dan data testing
X = dataset.drop('target', axis=1)
y = dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melakukan training model
model = LogisticRegression()
model.fit(X_train, y_train)

# Definisikan endpoint untuk menerima request POST
@app.route('/predict', methods=['POST'])
def predict():
    # Mengambil data input dari request
    data = request.get_json()

    # Melakukan preprocessing pada data input
    # Misalnya, jika data input berupa array numerik,  dapat mengubahnya menjadi numpy array
    input_data = np.array(data)

    # Melakukan prediksi menggunakan model yang telah di-train sebelumnya
    prediction = model.predict(input_data)

    # Mengubah hasil prediksi menjadi format JSON
    result = {'prediction': prediction}

    # Mengembalikan hasil prediksi dalam format JSON
    return jsonify(result)

# Menjalankan aplikasi Flask di localhost dengan port 5000
if __name__ == '__main__':
    app.run(port=5000)
