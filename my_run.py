import cv2
import torch
from depth_anything.dpt import DepthAnything
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import numpy as np
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = 'vits'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

# Função para capturar o vídeo da webcam
def capturar_video():
    # Abrir a webcam
    cap = cv2.VideoCapture(0)

    # Verificar se a webcam foi aberta corretamente
    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        return

    # Loop para capturar o vídeo da webcam
    while True:
        # Capturar o frame da webcam
        ret, image = cap.read()
        image = cv2.resize(image, dsize = [640, 480])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]
        
        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        
        depth = depth.cpu().numpy().astype(np.uint8)

        print(depth.min(), depth.max())

        #_, depth = cv2.threshold(depth, 120, 255, cv2.THRESH_BINARY)


        # Mostrar o frame capturado
        cv2.imshow('Webcam', depth)

        # Verificar se a tecla 'q' foi pressionada para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a webcam e fechar todas as janelas
    cap.release()
    cv2.destroyAllWindows()

# Chamar a função para capturar o vídeo
capturar_video()
