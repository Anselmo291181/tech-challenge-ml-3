import os
import random
import pandas as pd

def generate_medical_reports(num_samples: int = 2200, output_path: str = "data/laudos_medicos.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    normal_templates = [
        "Exame de tórax dentro dos padrões da normalidade. Sem consolidações parenquimatosas.",
        "Paciente assintomático. Estruturas ósseas e tecidos moles sem alterações.",
        "Laudo de EKG normal. Ritmo sinusal regular sem sinais de isquemia aguda.",
        "Tomografia de crânio sem achados patológicos significativos. Ventrículos normais.",
        "Ultrassom abdominal total sem anormalidades detectadas. Órgãos com dimensões preservadas."
    ]
    
    atencao_templates = [
        "Sinais sutis de opacidade focal em base pulmonar direita. Recomendada reavaliação.",
        "Paciente refere dor abdominal moderada. Pequena alteração nas enzimas hepáticas.",
        "Alteração na repolarização ventricular em parede inferior. Monitorar sintomas.",
        "Nódulo tireoidiano indeterminado de 1.2cm. Sugere-se acompanhamento ultrassonográfico.",
        "Pequeno derrame pleural bilateral sem sinais de consolidação grave."
    ]
    
    urgente_templates = [
        "Sinais claros de pneumotórax hipertensivo à esquerda. Necessita intervenção imediata.",
        "Infarto agudo do miocárdio com supra de ST detectado em parede anterior.",
        "Hemorragia intraparenquimatosa volumosa em região temporal esquerda com desvio de linha média.",
        "Suspeita grave de embolia pulmonar massiva. Angiotomografia urgente necessária.",
        "Ruptura de aneurisma de aorta abdominal identificada. Encaminhar imediatamente ao centro cirúrgico."
    ]
    
    data = []
    for _ in range(num_samples):
        urgencia = random.choices(["normal", "atenção", "urgente"], weights=[0.5, 0.3, 0.2])[0]
        if urgencia == "normal":
            texto = random.choice(normal_templates)
        elif urgencia == "atenção":
            texto = random.choice(atencao_templates)
        else:
            texto = random.choice(urgente_templates)
            
        data.append({"laudo_medico": texto, "urgencia": urgencia})
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Dataset sintético com {len(df)} amostras gerado com sucesso em: {output_path}")

if __name__ == "__main__":
    generate_medical_reports()