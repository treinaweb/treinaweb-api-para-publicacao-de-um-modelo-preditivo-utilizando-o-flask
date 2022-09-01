import pickle
from flask import Flask, request
import numpy as np

with open('modelo/treino.pkl', 'rb') as modelo_arquivo:
    modelo = pickle.load(modelo_arquivo)

app = Flask(__name__)

@app.route('/predict', methods=["GET","POST"])
def predict():
    request_data = request.get_json()
    s_length = request_data["s_length"]
    s_width = request_data["s_width"]
    p_length = request_data["p_length"]
    p_width = request_data["p_width"]

    previsao = modelo.predict(np.array(
        [[
            s_length,
            s_width,
            p_length,
            p_width
        ]]
    ))

    return str(previsao)
