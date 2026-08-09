import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Configuração do MediaPipe para Face Mesh (rastreamento dos olhos)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Pega a resolução da tela
screen_w, screen_h = pyautogui.size()

def run_virtual_mouse():
    cap = cv2.VideoCapture(0)
    print("Iniciando controle do Mouse Virtual via Eye Tracking...")
    print("Pressione 'q' para sair.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Espelha o frame para não inverter o movimento
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa o rosto
        output = face_mesh.process(rgb_frame)
        if output.multi_face_landmarks:
            landmarks = output.multi_face_landmarks[0].landmark
            
            # Ponto 473 é o centro da íris do olho esquerdo
            # Pode-se usar a média entre o 473 e o olho direito para mais precisão
            left_eye = landmarks[473] 
            
            # Converte as coordenadas relativas (0 a 1) para a resolução da tela
            # Multiplicamos por um fator de sensibilidade para não precisar virar muito a cabeça/olho
            x = int(left_eye.x * screen_w * 1.5 - (screen_w * 0.25))
            y = int(left_eye.y * screen_h * 1.5 - (screen_h * 0.25))
            
            # Move o cursor suavemente (ignorar limites)
            try:
                pyautogui.moveTo(x, y, duration=0.1)
            except Exception:
                pass
                
            # TODO: Lógica de Blink (piscar) para clicar
            # left_eye_top = landmarks[159]
            # left_eye_bottom = landmarks[145]
            # if abs(left_eye_top.y - left_eye_bottom.y) < 0.005:
            #     pyautogui.click()
                
        cv2.imshow("IrisFlow Virtual Mouse (Backend)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_virtual_mouse()
