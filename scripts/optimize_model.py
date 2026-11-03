import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

def convert_to_onnx(model_path: str = "models/model.joblib", onnx_path: str = "models/model.onnx"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Execute o treino primeiro.")
        
    pipeline = joblib.load(model_path)
    initial_type = [("string_input", StringTensorType([None, 1]))]
    
    print("Convertendo pipeline Scikit-Learn para formato ONNX...")
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"Modelo otimizado salvo com sucesso em: {onnx_path}")

if __name__ == "__main__":
    convert_to_onnx()