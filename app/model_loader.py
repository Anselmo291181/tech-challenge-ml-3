import os
import joblib
import numpy as np
import onnxruntime as rt

class InferenceEngine:
    def __init__(self, joblib_path: str = "models/model.joblib", onnx_path: str = "models/model.onnx"):
        self.joblib_path = joblib_path
        self.onnx_path = onnx_path
        self.use_onnx = False
        self.sk_pipeline = None
        self.onnx_session = None
        self.load_models()

    def load_models(self):
        if os.path.exists(self.onnx_path):
            self.onnx_session = rt.InferenceSession(self.onnx_path)
            self.input_name = self.onnx_session.get_inputs()[0].name
            self.output_label = self.onnx_session.get_outputs()[0].name
            self.output_proba = self.onnx_session.get_outputs()[1].name
            self.use_onnx = True
            print("Engine de Inferência: ONNX Runtime carregado.")
        elif os.path.exists(self.joblib_path):
            self.sk_pipeline = joblib.load(self.joblib_path)
            print("Engine de Inferência: Scikit-Learn Joblib carregado.")
        else:
            print("Aviso: Nenhum modelo pré-treinado encontrado.")

    def predict(self, text: str):
        if self.use_onnx and self.onnx_session:
            input_data = np.array([[text]], dtype=object)
            res = self.onnx_session.run([self.output_label, self.output_proba], {self.input_name: input_data})
            label = str(res[0][0])
            probas = res[1][0]
            confidence = float(max(probas.values()))
            return label, confidence, "ONNXRuntime"
        elif self.sk_pipeline:
            proba = self.sk_pipeline.predict_proba([text])[0]
            label = self.sk_pipeline.predict([text])[0]
            confidence = float(np.max(proba))
            return label, confidence, "Scikit-Learn"
        else:
            raise RuntimeError("Nenhum modelo carregado para inferência.")