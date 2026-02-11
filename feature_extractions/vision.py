import numpy as np
from scipy import ndimage
import cv2
from skimage.feature import local_binary_pattern  #OTIMIZADO: LBP rápido
from skimage import filters as skfilters  #OTIMIZADO: filtros rápidos
from dataclasses import dataclass
from typing import List, Callable


@dataclass
class NodeFunction:
    func: Callable
    name: str
    n_inputs: int


def vision_functions() -> List[NodeFunction]:
        """Initialize the set of possible node functions"""
        
        def conv3x3(x: np.ndarray, param: float) -> np.ndarray:
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            # Handle each channel separately
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                kernel = np.array([[param, param, param],
                                 [param, 1.0, param],
                                 [param, param, param]])
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.convolve(x[..., c], kernel, mode='reflect')
                return result
            
            kernel = np.array([[param, param, param],
                                [param, 1.0, param],
                                [param, param, param]])
            result = ndimage.convolve(x, kernel, mode='reflect')
            return result
            
        def max_pool(x: np.ndarray, param: float) -> np.ndarray:
            size = max(2, int(abs(param * 3)))
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.maximum_filter(x, size=size)
                return result
            return ndimage.maximum_filter(x, size=size)
            
        def avg_pool(x: np.ndarray, param: float) -> np.ndarray:
            size = max(2, int(abs(param * 3)))
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.uniform_filter(x, size=size)
                return result
            return ndimage.uniform_filter(x, size=size)
        
        def gaussian_blur(x: np.ndarray, param: float) -> np.ndarray:
            sigma = abs(param)
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.gaussian_filter(x, sigma=sigma)
                return result
            return ndimage.gaussian_filter(x, sigma=sigma)
            
        def sobel_x(x: np.ndarray, param: float) -> np.ndarray:
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.sobel(x, axis=0) * param
                return result
            return ndimage.sobel(x, axis=0) * param
            
        def sobel_y(x: np.ndarray, param: float) -> np.ndarray:
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.sobel(x, axis=1) * param
                return result
            return ndimage.sobel(x, axis=1) * param
            
        def threshold(x: np.ndarray, param: float) -> np.ndarray:
            return np.where(x > param, 1.0, 0.0)
            
        def normalize(x: np.ndarray, param: float) -> np.ndarray:
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    channel = x[..., c]
                    min_val = np.min(channel)
                    max_val = np.max(channel)
                    if max_val > min_val:
                        result[..., c] = (channel - min_val) / (max_val - min_val)
                    else:
                        result[..., c] = channel
                return result
            else:
                min_val = np.min(x)
                max_val = np.max(x)
                if max_val > min_val:
                    return (x - min_val) / (max_val - min_val)
                return x
            
        def subtract(x1: np.ndarray, x2: np.ndarray, param: float) -> np.ndarray:
            # Ensure the arrays have the same shape
            if x1.shape != x2.shape:
                x2 = cv2.resize(x2, (x1.shape[1], x1.shape[0]))
                if len(x1.shape) == 3 and len(x2.shape) == 2:
                    x2 = np.expand_dims(x2, axis=-1)
                    x2 = np.repeat(x2, x1.shape[2], axis=-1)
            return x1 - x2 * param
        
        def add(x1: np.ndarray, x2: np.ndarray, param: float) -> np.ndarray:
            # Ensure the arrays have the same shape
            if x1.shape != x2.shape:
                x2 = cv2.resize(x2, (x1.shape[1], x1.shape[0]))
                if len(x1.shape) == 3 and len(x2.shape) == 2:
                    x2 = np.expand_dims(x2, axis=-1)
                    x2 = np.repeat(x2, x1.shape[2], axis=-1)
            return x1 + x2 * param
            
        def multiply(x1: np.ndarray, x2: np.ndarray, param: float) -> np.ndarray:
            # Ensure the arrays have the same shape
            if x1.shape != x2.shape:
                x2 = cv2.resize(x2, (x1.shape[1], x1.shape[0]))
                if len(x1.shape) == 3 and len(x2.shape) == 2:
                    x2 = np.expand_dims(x2, axis=-1)
                    x2 = np.repeat(x2, x1.shape[2], axis=-1)
            return x1 * x2 * param
        
        def laplacian(x: np.ndarray, param: float) -> np.ndarray:
            """
            Calcula o Laplaciano usando diferenças finitas.
            """
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            laplacian = np.zeros_like(x)
            
            # Calcula segundas derivadas em x e y
            laplacian[1:-1, 1:-1] = (
                x[1:-1, 2:] +    # direita
                x[1:-1, :-2] +   # esquerda
                x[2:, 1:-1] +    # baixo
                x[:-2, 1:-1] -   # cima
                4 * x[1:-1, 1:-1] # centro
            )
            
            return laplacian
        
        def roberts(x: np.ndarray, param: float) -> np.ndarray:
            """
            Implementa o operador de Roberts para detecção de bordas.
            """
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            # Kernels de Roberts
            roberts_cross_v = np.array([[ 0, 0, 0],
                                    [ 0, 1, 0],
                                    [ 0, 0,-1]])
            
            roberts_cross_h = np.array([[ 0, 0, 0],
                                    [ 0, 0, 1],
                                    [ 0,-1, 0]])
            
            # Garantir que a imagem é float
            x = x.astype('float64')
            vertical = ndimage.convolve(x, roberts_cross_v)
            horizontal = ndimage.convolve(x, roberts_cross_h)
            
            # Magnitude do gradiente
            edges = np.sqrt(np.square(horizontal) + np.square(vertical))
            
            max_val = np.max(edges)
            if max_val > 0:
                # Normalizar para 0-255
                edges = edges * 255.0 / max_val                
                return edges.astype(np.uint8)
            return x
        
        def median_blur(x: np.ndarray, param: float) -> np.ndarray:
            """
            Aplica filtro de mediana em uma imagem.
            Usa OpenCV que é muito mais rápido que loops Python.
            """
            param = max(3, int(abs(param)))
            # OpenCV requer tamanho ímpar
            if param % 2 == 0:
                param += 1
            
            # Converter para uint8 para OpenCV (evita warning de cast)
            x_clipped = np.clip(x, 0, 255)
            x_uint8 = np.round(x_clipped).astype(np.uint8)
            
            if len(x.shape) == 3:
                result = np.zeros_like(x, dtype=np.float64)
                for c in range(x.shape[2]):
                    result[..., c] = cv2.medianBlur(x_uint8[..., c], param)
                return result
            return cv2.medianBlur(x_uint8, param).astype(np.float64)
        
        def sharpen(x: np.ndarray, param: float) -> np.ndarray:
            """
            Sharpening com controle de intensidade.
            
            Args:
                x: Imagem de entrada
                param: Intensidade do sharpening (1.0 é normal)
            """
            # Kernel Laplaciano
            kernel = np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ])
            
            # Calcula bordas
            edges = ndimage.convolve(x, kernel)
            
            # Aplica sharpening com intensidade ajustável
            sharpened = x + (edges * param)
            
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        
        def grayscale(x: np.ndarray, param: float) -> np.ndarray:
            """
            Converte imagem RGB para escala de cinza.
            Reduz dimensionalidade e estabiliza o grafo CGP.
            """
            if len(x.shape) == 3 and x.shape[2] >= 3:
                # Pesos padrão para conversão RGB->cinza (luminosidade)
                gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
                return gray
            return x
        
        def sobel_magnitude(x: np.ndarray, param: float) -> np.ndarray:
            """
            Calcula magnitude do gradiente usando Sobel.
            magnitude = sqrt(sobel_x² + sobel_y²)
            """
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)[..., :1]  # Manter 1 canal
                for c in range(min(x.shape[2], 3)):
                    sx = ndimage.sobel(x[..., c], axis=0)
                    sy = ndimage.sobel(x[..., c], axis=1)
                    result[..., 0] = np.sqrt(sx**2 + sy**2)
                return result[..., 0]
            sx = ndimage.sobel(x, axis=0)
            sy = ndimage.sobel(x, axis=1)
            return np.sqrt(sx**2 + sy**2)
        
        def local_variance(x: np.ndarray, param: float) -> np.ndarray:
            """
            Calcula variância local usando janela de tamanho baseado em param.
            Bom para detecção de textura.
            """
            size = max(3, int(abs(param) * 5) | 1)  # Garante ímpar
            if size > x.shape[0] or size > x.shape[1]:
                size = min(x.shape[0], x.shape[1]) | 1
            
            # Usar generic_filter para variância
            # Variância = mean(x²) - mean(x)²
            mean_x = ndimage.uniform_filter(x, size=size)
            mean_x2 = ndimage.uniform_filter(x**2, size=size)
            variance = mean_x2 - mean_x**2
            
            return variance
        
        def erosion(x: np.ndarray, param: float) -> np.ndarray:
            """
            Aplica erosão morfológica.
            """
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            size = max(3, int(abs(param) * 3) | 1)
            structure = np.ones((size, size))
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    binary = x[..., c] > 127
                    eroded = ndimage.binary_erosion(binary, structure)
                    result[..., c] = eroded.astype(x.dtype)
                return result
            binary = x > 127
            eroded = ndimage.binary_erosion(binary, structure)
            return eroded.astype(x.dtype)
        
        def dilation(x: np.ndarray, param: float) -> np.ndarray:
            """
            Aplica dilatação morfológica.
            """
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            size = max(3, int(abs(param) * 3) | 1)
            structure = np.ones((size, size))
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    binary = x[..., c] > 127
                    dilated = ndimage.binary_dilation(binary, structure)
                    result[..., c] = dilated.astype(x.dtype)
                return result
            binary = x > 127
            dilated = ndimage.binary_dilation(binary, structure)
            return dilated.astype(x.dtype)
        
        def opening(x: np.ndarray, param: float) -> np.ndarray:
            """
            Abertura morfológica (erosão + dilatação).
            Remove ruído pequeno preservando formas.
            """
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            size = max(3, int(abs(param) * 3) | 1)
            structure = np.ones((size, size))
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    binary = x[..., c] > 127
                    eroded = ndimage.binary_erosion(binary, structure)
                    result[..., c] = ndimage.binary_dilation(eroded, structure).astype(x.dtype)
                return result
            binary = x > 127
            eroded = ndimage.binary_erosion(binary, structure)
            return ndimage.binary_dilation(eroded, structure).astype(x.dtype)
        
        def log_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Laplaciano de Gaussiana (LoG).
            Detecta bordas e regiões de alta frequência.
            """
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            sigma = max(0.5, abs(param) * 2)
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.gaussian_laplace(x[..., c], sigma=sigma)
                return result
            return ndimage.gaussian_laplace(x, sigma=sigma)
        
        def dog_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Diferença de Gaussianas (DoG).
            Aproximação do centro-surround da visão biológica.
            """
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            sigma1 = max(0.5, abs(param) * 1)
            sigma2 = sigma1 * 1.5
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    g1 = ndimage.gaussian_filter(x[..., c], sigma=sigma1)
                    g2 = ndimage.gaussian_filter(x[..., c], sigma=sigma2)
                    result[..., c] = g1 - g2
                return result
            g1 = ndimage.gaussian_filter(x, sigma=sigma1)
            g2 = ndimage.gaussian_filter(x, sigma=sigma2)
            return g1 - g2
        
        def lbp(x: np.ndarray, param: float) -> np.ndarray:
            """
            Local Binary Pattern (LBP) - OTIMIZADO com skimage.
            Muito relevante para textura de pelagem/rosto.
            """
            radius = max(1, int(abs(param) * 2))
            points = 8
            
            # Garantir grayscale
            if len(x.shape) == 3:
                gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
            else:
                gray = x
            
            # Converter para uint8 para evitar warning do skimage
            gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
            
            # skimage LBP é muito mais rápido (C-optimized)
            lbp_result = local_binary_pattern(gray_uint8, P=points, R=radius, method='uniform')
            
            return lbp_result.astype(np.float64)
        
        def canny_edges(x: np.ndarray, param: float) -> np.ndarray:
            """
            Detecção de bordas usando Canny - OTIMIZADO com OpenCV.
            Much faster than Python loops.
            """
            # Converter para grayscale se necessário
            if len(x.shape) == 3:
                gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
            else:
                gray = x.copy()
            
            # Garantir uint8 para OpenCV
            gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
            
            # Parâmetros de limiar baseados em param
            low_threshold = max(10, int(abs(param) * 50))
            high_threshold = max(20, low_threshold * 2)
            
            # cv2.Canny é muito mais rápido (C-optimized)
            edges = cv2.Canny(gray_uint8, low_threshold, high_threshold)
            
            return edges.astype(np.float64)
        
        def scharr_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Operador de Scharr (similar ao Sobel, melhor aproximação).
            Kernels 3×3 que aproximam melhor a derivada.
            """
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            # Kernels Scharr
            scharr_x = np.array([[-3, 0, 3],
                                [-10, 0, 10],
                                [-3, 0, 3]])
            scharr_y = np.array([[-3, -10, -3],
                                [0, 0, 0],
                                [3, 10, 3]])
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    sx = ndimage.convolve(x[..., c], scharr_x, mode='reflect')
                    sy = ndimage.convolve(x[..., c], scharr_y, mode='reflect')
                    if abs(param) > 0.5:
                        result[..., c] = np.sqrt(sx**2 + sy**2)
                    else:
                        result[..., c] = sx * abs(param) + sy * abs(param)
                return result
            sx = ndimage.convolve(x, scharr_x, mode='reflect')
            sy = ndimage.convolve(x, scharr_y, mode='reflect')
            return np.sqrt(sx**2 + sy**2)
        
        def prewitt_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Operador de Prewitt (outro detector de gradiente).
            Alternativa ao Sobel/Scharr.
            """
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            prewitt_x = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
            prewitt_y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    sx = ndimage.convolve(x[..., c], prewitt_x, mode='reflect')
                    sy = ndimage.convolve(x[..., c], prewitt_y, mode='reflect')
                    if abs(param) > 0.5:
                        result[..., c] = np.sqrt(sx**2 + sy**2)
                    else:
                        result[..., c] = sx + sy
                return result
            sx = ndimage.convolve(x, prewitt_x, mode='reflect')
            sy = ndimage.convolve(x, prewitt_y, mode='reflect')
            return np.sqrt(sx**2 + sy**2)
        
        def gabor_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Filtro de Gabor (orientado + frequência).
            Bom para textura e orientação.
            O parâmetro controla ângulo (0-π) e escala.
            """
            # Garantir que a entrada é numpy array float
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            # Parâmetros baseados em param
            theta = (abs(param) % 1) * np.pi  # Ângulo entre 0 e π
            sigma = max(1.0, abs(param) * 2)  # Escala
            frequency = max(0.1, 1.0 / (sigma * 2))  # Frequência
            
            # Criar kernel Gabor
            kernel_size = int(max(3, sigma * 3)) | 1  # Garante ímpar
            half_size = kernel_size // 2
            
            # Gerar coordenadas
            y, x_coord = np.meshgrid(
                np.arange(-half_size, half_size + 1),
                np.arange(-half_size, half_size + 1)
            )
            
            # Rotacionar coordenadas
            x_rot = x_coord * np.cos(theta) + y * np.sin(theta)
            y_rot = -x_coord * np.sin(theta) + y * np.cos(theta)
            
            # Kernel Gabor: Gaussiana * Cossenóide
            gaussian = np.exp(-(x_rot**2 + y_rot**2) / (2 * sigma**2))
            carrier = np.exp(1j * 2 * np.pi * frequency * x_rot)
            kernel = gaussian * carrier
            
            # Aplicar em cada canal
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    # Parte real do Gabor
                    result[..., c] = ndimage.convolve(x[..., c], np.real(kernel), mode='reflect')
                return result
            return ndimage.convolve(x, np.real(kernel), mode='reflect')
        
        def local_entropy(x: np.ndarray, param: float) -> np.ndarray:
            """
            Calcula variância local (proxy rápido para entropia).
            Usa filtros uniformes do scipy que são vetorizados.
            """
            size = max(3, int(abs(param) * 3) | 1)
            if size > x.shape[0] or size > x.shape[1]:
                size = min(x.shape[0], x.shape[1]) | 1
            
            # Variância como proxy para "desordem" local
            mean_x = ndimage.uniform_filter(x.astype(np.float64), size=size)
            mean_x2 = ndimage.uniform_filter(x.astype(np.float64)**2, size=size)
            variance = mean_x2 - mean_x**2
            
            # Garantir valores não-negativos para log
            variance = np.abs(variance)
            
            # Adicionar epsilon para evitar log de zero e usar nan_to_num
            return np.nan_to_num(np.log(variance + 1e-6), nan=0.0, posinf=0.0, neginf=0.0)
        
        def select_channel(x: np.ndarray, param: float) -> np.ndarray:
            """
            Seleciona um canal específico da imagem.
            param < -0.33: Canal R (0)
            param > 0.33: Canal G (1)
            else: Canal B (2)
            """
            if len(x.shape) != 3 or x.shape[2] < 3:
                return x
            
            if param < -0.33:
                return x[..., 0]  # Red
            elif param > 0.33:
                return x[..., 1]  # Green
            else:
                return x[..., 2]  # Blue
        
        def hsv_channel(x: np.ndarray, param: float) -> np.ndarray:
            """
            Converte RGB para HSV e retorna canal configurável.
            Permite selecionar H, S ou V baseado em param.
            
            param < -0.33: Retorna H (Hue/Matiz)
            -0.33 <= param <= 0.33: Retorna S (Saturação)
            param > 0.33: Retorna V (Valor/Brilho)
            """
            if len(x.shape) != 3 or x.shape[2] < 3:
                return x
            
            # Converter para HSV usando OpenCV
            if x.dtype != np.uint8:
                x_uint8 = np.clip(x, 0, 255).astype(np.uint8)
            else:
                x_uint8 = x
            
            hsv = cv2.cvtColor(x_uint8, cv2.COLOR_RGB2HSV)
            
            # Selecionar canal baseado em param
            if param < -0.33:
                # Canal H (Hue) - normalizado para 0-1
                return (hsv[..., 0] / 180.0).astype(np.float64)
            elif param > 0.33:
                # Canal V (Valor/Brilho) - normalizado para 0-1
                return (hsv[..., 2] / 255.0).astype(np.float64)
            else:
                # Canal S (Saturação) - normalizado para 0-1
                return (hsv[..., 1] / 255.0).astype(np.float64)
        
        def rgb_channel(x: np.ndarray, param: float) -> np.ndarray:
            """
            Seleciona canal RGB específico da imagem.
            Útil para focar em um canal de cor.
            
            param < -0.33: Retorna R (Vermelho)
            -0.33 <= param <= 0.33: Retorna G (Verde)
            param > 0.33: Retorna B (Azul)
            """
            if len(x.shape) != 3 or x.shape[2] < 3:
                return x
            
            if param < -0.33:
                return x[..., 0]  # Red
            elif param > 0.33:
                return x[..., 2]  # Blue
            else:
                return x[..., 1]  # Green
        
        def channel_difference(x: np.ndarray, param: float) -> np.ndarray:
            """
            Calcula diferença entre canais RGB.
            param < -0.33: R - G
            param > 0.33: G - B
            else: R - B
            Realça diferenças de cor.
            """
            if len(x.shape) != 3 or x.shape[2] < 3:
                return x
            
            if param < -0.33:
                # R - G
                return x[..., 0] - x[..., 1]
            elif param > 0.33:
                # G - B
                return x[..., 1] - x[..., 2]
            else:
                # R - B
                return x[..., 0] - x[..., 2]
        
        def closing(x: np.ndarray, param: float) -> np.ndarray:
            """
            Fechamento morfológico (dilatação + erosão).
            Preenche buracos pequenos e conecta regiões.
            """
            x = np.asarray(x)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            size = max(3, int(abs(param) * 3) | 1)
            structure = np.ones((size, size))
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    binary = x[..., c] > 127
                    dilated = ndimage.binary_dilation(binary, structure)
                    result[..., c] = ndimage.binary_erosion(dilated, structure).astype(x.dtype)
                return result
            binary = x > 127
            dilated = ndimage.binary_dilation(binary, structure)
            return ndimage.binary_erosion(dilated, structure).astype(x.dtype)
        
        def morphological_gradient(x: np.ndarray, param: float) -> np.ndarray:
            """
            Gradiente morfológico: dilatação - erosão.
            Destaca contornos de objetos.
            """
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            size = max(3, int(abs(param) * 3) | 1)
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    dilated = ndimage.grey_dilation(x[..., c], size=size)
                    eroded = ndimage.grey_erosion(x[..., c], size=size)
                    result[..., c] = dilated - eroded
                return result
            dilated = ndimage.grey_dilation(x, size=size)
            eroded = ndimage.grey_erosion(x, size=size)
            return dilated - eroded
        
        def bilateral_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Filtro bilateral simplificado (versão rápida).
            Usa gaussian + preserva bordas por diferença.
            """
            x = np.asarray(x, dtype=np.float64)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            if x.ndim > 3:
                x = x[..., :3]
            
            sigma = max(0.5, abs(param) * 3)
            
            # Suavização Gaussiana rápida
            blurred = ndimage.gaussian_filter(x.astype(np.float64), sigma=sigma)
            
            # Detecção de bordas (gradiente)
            grad_mag = np.sqrt(ndimage.sobel(x, axis=0)**2 + ndimage.sobel(x, axis=1)**2)
            
            # Preservar bordas: blended = blurred onde suave, x onde borda
            edge_mask = np.exp(-grad_mag / 50)  # Maior borda = menor peso
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = blurred[..., c] * edge_mask + x[..., c] * (1 - edge_mask)
                return result
            return blurred * edge_mask + x * (1 - edge_mask)
        
        def min_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Filtro de mínimo local.
            Complemento do max_pool para análises de intensidade.
            """
            size = max(2, int(abs(param * 3)))
            if size % 2 == 0:
                size += 1
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.minimum_filter(x, size=size)
                return result
            return ndimage.minimum_filter(x, size=size)
        
        def percentile_filter(x: np.ndarray, param: float) -> np.ndarray:
            """
            Filtro de percentil local.
            param < 0: percentil baixo (p25)
            param >= 0: percentil alto (p75)
            """
            size = max(3, int(abs(param) * 5) | 1)
            percentile = 25 if param < 0 else 75
            
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.percentile_filter(
                        x[..., c], percentile, size=size
                    )
                return result
            return ndimage.percentile_filter(x, percentile, size=size)
        
        def abs_value(x: np.ndarray, param: float) -> np.ndarray:
            """
            Valor absoluto (np.abs).
            Útil após subtração ou Sobel para não perder sinal.
            """
            return np.abs(x)
        
        def clip_normalize(x: np.ndarray, param: float) -> np.ndarray:
            """
            Clip e normaliza para intervalo [0, 1].
            Garante saída controlada antes do próximo nó.
            """
            clipped = np.clip(x, 0, 255)
            max_val = np.max(clipped)
            if max_val > 0:
                return clipped / max_val
            return clipped
        
        def power_transform(x: np.ndarray, param: float) -> np.ndarray:
            """
            Transformação de potência (gamma).
            Realça sombras (gamma < 1) ou realces (gamma > 1).
            """
            gamma = max(0.1, abs(param) * 3)
            x_normalized = np.clip(x, 0, 255) / 255.0
            result = np.power(x_normalized, gamma) * 255
            return np.nan_to_num(result, nan=0.0, posinf=255.0, neginf=0.0)
        
        def contrast_stretch(x: np.ndarray, param: float) -> np.ndarray:
            """
            Stretch de contraste por percentis (2% e 98%).
            Mais robusto a outliers que min/max.
            """
            p_low = max(1, int(abs(param) * 5))
            p_high = 100 - p_low
            
            low = np.percentile(x, p_low)
            high = np.percentile(x, p_high)
            
            if high > low:
                return (x - low) / (high - low) * 255
            return x

        return [
            NodeFunction(conv3x3, "conv3x3", 1),
            NodeFunction(max_pool, "max_pool", 1),
            NodeFunction(avg_pool, "avg_pool", 1),
            NodeFunction(gaussian_blur, "gaussian_blur", 1),
            NodeFunction(sobel_x, "sobel_x", 1),
            NodeFunction(sobel_y, "sobel_y", 1),
            NodeFunction(sobel_magnitude, "sobel_magnitude", 1),  # NOVA
            NodeFunction(threshold, "threshold", 1),
            NodeFunction(normalize, "normalize", 1),
            NodeFunction(add, "add", 2),
            NodeFunction(subtract, "subtract", 2),
            NodeFunction(laplacian, "laplacian", 1),
            NodeFunction(roberts, "roberts", 1),
            NodeFunction(median_blur, "median_blur", 1),
            NodeFunction(sharpen, "sharpen", 1),
            NodeFunction(multiply, "multiply", 2),
            # Novas funções adicionadas:
            NodeFunction(grayscale, "grayscale", 1),           # NOVA
            NodeFunction(local_variance, "local_variance", 1), # NOVA
            NodeFunction(erosion, "erosion", 1),               # NOVA
            NodeFunction(dilation, "dilation", 1),            # NOVA
            NodeFunction(opening, "opening", 1),              # NOVA
            NodeFunction(log_filter, "log_filter", 1),        # NOVA
            NodeFunction(dog_filter, "dog_filter", 1),        # NOVA
            NodeFunction(lbp, "lbp", 1),                      # NOVA
            NodeFunction(canny_edges, "canny_edges", 1),      # NOVA
            NodeFunction(scharr_filter, "scharr_filter", 1), # NOVA
            NodeFunction(prewitt_filter, "prewitt_filter", 1), # NOVA
            # Novas funções de textura e cor:
            NodeFunction(gabor_filter, "gabor_filter", 1),           # NOVA
            NodeFunction(local_entropy, "local_entropy", 1),         # NOVA
            NodeFunction(rgb_channel, "rgb_channel", 1),               # NOVA
            NodeFunction(hsv_channel, "hsv_channel", 1),             # NOVA
            NodeFunction(channel_difference, "channel_difference", 1), # NOVA
            # Novas funções morfológicas e estatísticas:
            NodeFunction(closing, "closing", 1),                   # NOVA
            NodeFunction(morphological_gradient, "morphological_gradient", 1), # NOVA
            NodeFunction(bilateral_filter, "bilateral_filter", 1), # NOVA
            NodeFunction(min_filter, "min_filter", 1),             # NOVA
            NodeFunction(percentile_filter, "percentile_filter", 1), # NOVA
            NodeFunction(abs_value, "abs_value", 1),             # NOVA
            NodeFunction(clip_normalize, "clip_normalize", 1),   # NOVA
            NodeFunction(power_transform, "power_transform", 1), # NOVA
            NodeFunction(contrast_stretch, "contrast_stretch", 1) # NOVA
        ]