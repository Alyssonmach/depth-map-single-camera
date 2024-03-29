# Importação dos pacotes utilitários
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything
from torchvision.transforms import Compose
import torch.nn.functional as F
import numpy as np
import torch
import cv2

# Verificando se há uma GPU disponível com o PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Escolhido o codificador adequado (na ordem crescente de robustez)
encoders = ['vits', 'vitb', 'vitl']
# Carregando o modelo pré-treinado
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoders[2])).to(DEVICE).eval()

# Organizando a função que processa a imagem adequada para os padrões do modelo
transform = Compose([
        Resize(
            width = 518, height = 518,
            resize_target = False, keep_aspect_ratio = True,
            ensure_multiple_of = 14, resize_method = 'lower_bound',
            image_interpolation_method = cv2.INTER_CUBIC,),
        NormalizeImage(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

# Captura um vídeo de entrada através de uma foto externa
def main(webcam = True, video_path = ''):

    # Conecta a webcam para realizar inferências
    if webcam: cap = cv2.VideoCapture(0)
    # Conecta a uma saída de vídeo para realizar inferências 
    else: cap = cv2.VideoCapture(video_path)

    # Verificar se conseguimos capturar o vídeo de entrada corretamente
    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        return

    # Loop para capturar o vídeo de entrada frame a frame
    while True:
        # Intera sobre o próximo frame do vídeo de entrada
        ret, image = cap.read()
        # Redimesiona o frame atual 
        image = cv2.resize(image, dsize = [640, 480])
        # Pega uma cópia da imagem original de entrada
        image_original = image.copy()
        # Garante que o vídeo vai estar na escala RGB e normalizado
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # Obtém as dimensões do vídeo de entrada
        h, w = image.shape[:2]
        
        # Pré-processa o frame para realizar a inferência com o modelo
        image = transform({'image': image})['image']
        # Converte o frame atual em um objeto do PyTorch
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        # Realiza a estimação do mapa de profundidade no frame com o modelo
        with torch.no_grad(): depth = depth_anything(image)
        
        # Redimensiona a imagem de inferência para melhor visualização na plotagem
        depth = F.interpolate(depth[None], (h, w), mode = 'bilinear', align_corners = False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        # Converte de objeto PyTorch para um array do Numpy
        depth = depth.cpu().numpy().astype(np.uint8)
        # Converte a imagem de 1 canal para 3 canais RGB
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)

        # Coloca um título no frame original
        cv2.putText(img = image_original, text = 'Vídeo Original', org = (10, 30), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 255, 255), 
                    thickness = 2)
        
        # COloca um título no frame do mapa de profundidade
        cv2.putText(img = depth, text = 'Mapa de Profundidade', org = (10, 30), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255, 255, 255), 
                    thickness = 2)
        

        # Mostra na tela os resultados com o frame Atual
        cv2.imshow('Depth Anything', cv2.hconcat(src = [image_original, depth]))

        # Verifica se a tecla 'q' foi pressionada para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a webcam e fechar todas as janelas
    cap.release()
    cv2.destroyAllWindows()

# Chamar a função para inicializar a inferência em vídeo
if __name__ == "__main__":
    main()
