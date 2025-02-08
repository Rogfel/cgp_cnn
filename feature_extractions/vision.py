import numpy as np
from scipy import ndimage
import cv2
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
            if len(x.shape) == 3:
                result = np.zeros_like(x)
                for c in range(x.shape[2]):
                    result[..., c] = ndimage.sobel(x, axis=0) * param
                return result
            return ndimage.sobel(x, axis=0) * param
            
        def sobel_y(x: np.ndarray, param: float) -> np.ndarray:
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
            Para 2D: ∇²f = (∂²f/∂x²) + (∂²f/∂y²)
            """
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
            
            Args:
                image: Array NumPy 2D (imagem em escala de cinza)
            Returns:
                Imagem com bordas detectadas
            """
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
            
            Args:
                x: Array NumPy 2D (imagem)
                param: Tamanho da janela (deve ser ímpar)
            """
            param = int(param)
            # Verifica se kernel_size é ímpar
            if param % 2 == 0:
                param += 1
            pad = param // 2
            padded = np.pad(x, pad, mode='reflect')
            
            # Cria views deslizantes
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, 
                (param, param)
            )
            
            # Calcula mediana para cada janela
            result = np.median(windows, axis=(2,3))
            
            return result.astype(np.uint8)
        
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

        return [
            NodeFunction(conv3x3, "conv3x3", 1),
            NodeFunction(max_pool, "max_pool", 1),
            NodeFunction(avg_pool, "avg_pool", 1),
            NodeFunction(gaussian_blur, "gaussian_blur", 1),
            NodeFunction(sobel_x, "sobel_x", 1),
            NodeFunction(sobel_y, "sobel_y", 1),
            NodeFunction(threshold, "threshold", 1),
            NodeFunction(normalize, "normalize", 1),
            NodeFunction(add, "add", 2),
            NodeFunction(subtract, "subtract", 2),
            NodeFunction(laplacian, "laplacian", 1),
            NodeFunction(roberts, "roberts", 1),
            NodeFunction(median_blur, "median_blur", 1),
            NodeFunction(sharpen, "sharpen", 1),
            NodeFunction(multiply, "multiply", 2)
        ]