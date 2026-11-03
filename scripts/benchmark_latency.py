import time
import numpy as np
import joblib
import onnxruntime as rt

def benchmark():
    sample_text = "Paciente apresenta opacidade focal e dor torácica severa. Suspeita de complicação."
    iterations = 1000
    
    sk_model = joblib.load("models/model.joblib")
    sk_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = sk_model.predict([sample_text])
        sk_times.append((time.perf_counter() - t0) * 1000)
        
    sess = rt.InferenceSession("models/model.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    input_data = np.array([[sample_text]], dtype=object)
    
    onnx_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = sess.run([label_name], {input_name: input_data})
        onnx_times.append((time.perf_counter() - t0) * 1000)
        
    avg_sk = np.mean(sk_times)
    avg_onnx = np.mean(onnx_times)
    improvement = ((avg_sk - avg_onnx) / avg_sk) * 100
    
    print("\n" + "="*50)
    print("      BENCHMARK DE LATÊNCIA (1.000 INFERÊNCIAS)")
    print("="*50)
    print(f"Scikit-Learn / Joblib (Média): {avg_sk:.4f} ms por inferência")
    print(f"ONNX Runtime (Média):          {avg_onnx:.4f} ms por inferência")
    print(f"Ganho de Desempenho / Redução: {improvement:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    benchmark()