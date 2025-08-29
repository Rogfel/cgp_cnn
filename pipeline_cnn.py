# Pipeline ResNet implementation using conv_layers.py
import os
# Force CPU usage to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from feature_extractions import conv_layers as cnn
import keras
import tensorflow as tf
import numpy as np

# Configure TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

class ResNetPipeline:
    
    @staticmethod
    def residual_block(x, filters, kernel_size=3):
        """
        Implementa um bloco residual do ResNet
        """
        # Caminho principal (main path) - use 'same' padding to maintain spatial dimensions
        main_path = cnn.Layers.conv2D(x, filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        main_path = keras.layers.BatchNormalization()(main_path)
        main_path = keras.layers.ReLU()(main_path)
        
        main_path = cnn.Layers.conv2D(main_path, filters=filters, kernel_size=kernel_size, padding='same', activation=None)
        main_path = keras.layers.BatchNormalization()(main_path)
        
        # Caminho de atalho (shortcut path)
        if x.shape[-1] != filters:
            # Projeção para ajustar dimensões
            shortcut = cnn.Layers.conv2D(x, filters=filters, kernel_size=1, padding='same', activation=None)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = x
        
        # Adição dos caminhos
        output = keras.layers.Add()([main_path, shortcut])
        output = keras.layers.ReLU()(output)
        
        return output
    
    @staticmethod
    def bottleneck_block(x, filters, kernel_size=3):
        """
        Implementa um bloco bottleneck do ResNet (1x1, 3x3, 1x1)
        """
        # Caminho principal (1x1 -> 3x3 -> 1x1) - use 'same' padding
        main_path = cnn.Layers.conv2D(x, filters=filters//4, kernel_size=1, padding='same', activation=None)
        main_path = keras.layers.BatchNormalization()(main_path)
        main_path = keras.layers.ReLU()(main_path)
        
        main_path = cnn.Layers.conv2D(main_path, filters=filters//4, kernel_size=kernel_size, padding='same', activation=None)
        main_path = keras.layers.BatchNormalization()(main_path)
        main_path = keras.layers.ReLU()(main_path)
        
        main_path = cnn.Layers.conv2D(main_path, filters=filters, kernel_size=1, padding='same', activation=None)
        main_path = keras.layers.BatchNormalization()(main_path)
        
        # Caminho de atalho
        if x.shape[-1] != filters:
            shortcut = cnn.Layers.conv2D(x, filters=filters, kernel_size=1, padding='same', activation=None)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        else:
            shortcut = x
        
        # Adição dos caminhos
        output = cnn.Layers.summation(main_path, shortcut)
        output = keras.layers.ReLU()(output)
        
        return output
    
    @staticmethod
    def resnet_sequence(images_batch, num_classes=2, resnet_type='resnet18'):
        """
        Implementa a sequência completa do ResNet
        """
        # Camada inicial - rescaling expects a list, so we pass it correctly
        x = cnn.Layers.rescaling([images_batch])
        x = cnn.Layers.conv2D(x, filters=64, kernel_size=7, padding='same', activation=None)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = cnn.Layers.maxPool2D(x, pool_sizes=3, strides=2)
        
        if resnet_type == 'resnet18':
            # ResNet-18: 2 blocos residuais por estágio - adapted for 64x64 input
            x = ResNetPipeline.residual_block(x, filters=64)
            x = ResNetPipeline.residual_block(x, filters=64)
            
            # Reduzir dimensões antes do próximo estágio
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            x = ResNetPipeline.residual_block(x, filters=128)
            x = ResNetPipeline.residual_block(x, filters=128)
            
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            x = ResNetPipeline.residual_block(x, filters=256)
            x = ResNetPipeline.residual_block(x, filters=256)
            
            # Remove the last pooling layer to maintain larger spatial dimensions
            x = ResNetPipeline.residual_block(x, filters=512)
            x = ResNetPipeline.residual_block(x, filters=512)
            
        elif resnet_type == 'resnet34':
            # ResNet-34: 3-4-6-3 blocos residuais
            x = ResNetPipeline.residual_block(x, filters=64)
            x = ResNetPipeline.residual_block(x, filters=64)
            x = ResNetPipeline.residual_block(x, filters=64)
            
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            x = ResNetPipeline.residual_block(x, filters=128)
            x = ResNetPipeline.residual_block(x, filters=128)
            x = ResNetPipeline.residual_block(x, filters=128)
            x = ResNetPipeline.residual_block(x, filters=128)
            
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            for _ in range(6):
                x = ResNetPipeline.residual_block(x, filters=256)
            
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            for _ in range(3):
                x = ResNetPipeline.residual_block(x, filters=512)
                
        elif resnet_type == 'resnet50':
            # ResNet-50: usa blocos bottleneck
            x = ResNetPipeline.bottleneck_block(x, filters=256)
            x = ResNetPipeline.bottleneck_block(x, filters=256)
            x = ResNetPipeline.bottleneck_block(x, filters=256)
            
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            x = ResNetPipeline.bottleneck_block(x, filters=512)
            x = ResNetPipeline.bottleneck_block(x, filters=512)
            x = ResNetPipeline.bottleneck_block(x, filters=512)
            x = ResNetPipeline.bottleneck_block(x, filters=512)
            
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            for _ in range(6):
                x = ResNetPipeline.bottleneck_block(x, filters=1024)
            
            x = cnn.Layers.maxPool2D(x, pool_sizes=2, strides=2)
            for _ in range(3):
                x = ResNetPipeline.bottleneck_block(x, filters=2048)
        
        # Camadas finais - use adaptive pooling for smaller input sizes
        # For 64x64 input, we need to use the actual spatial dimensions
        x = keras.layers.GlobalAveragePooling2D()(x)  # This adapts to any spatial size
        x = cnn.Layers.flatten(x)
        
        # Camada de classificação
        x = keras.layers.Dense(num_classes, activation='softmax')(x)
        
        return x
    
    @staticmethod
    def create_resnet_model(input_shape=(224, 224, 3), num_classes=2, resnet_type='resnet18'):
        """
        Cria um modelo ResNet completo
        """
        inputs = keras.Input(shape=input_shape)
        outputs = ResNetPipeline.resnet_sequence(inputs, num_classes, resnet_type)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


def sequency_resnet(images_batch, resnet_type='resnet18'):
    """
    Função wrapper para compatibilidade com a estrutura existente
    """
    return ResNetPipeline.resnet_sequence(images_batch, resnet_type=resnet_type)


if __name__ == '__main__':
    from dataset import load
    from sklearn.model_selection import train_test_split
    
    # Configurar caminho do dataset
    load.PATH_DATASET = 'dataset/PetImages/'
    
    print('*****1/4 Load images dataset')
    
    # Carregar dataset usando a mesma estrutura do pipeline_cgp
    images, labels, classes = load.data(data_type='train')
    
    # Convert data from (batch, channels, height, width) to (batch, height, width, channels)
    # This is the format expected by Keras
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    
    # Criar modelo ResNet
    print('*****2/4 Create ResNet model')
    resnet_type = 'resnet18'  # Define the ResNet type
    model = ResNetPipeline.create_resnet_model(
        input_shape=(load.IMG_HEIGHT, load.IMG_WIDTH, 3), 
        num_classes=len(classes), 
        resnet_type=resnet_type
    )
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Mostrar resumo do modelo
    model.summary()
    
    print('*****3/4 Train ResNet model')
    
    # Treinar modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    print('*****4/4 Evaluate model')
    
    # Avaliar modelo
    test_loss, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {test_accuracy:.4f}")
    print(f"Validation loss: {test_loss:.4f}")
    
    # Save the model with ResNet type in the name
    model_filename = f"resnet_model_{resnet_type}.h5"
    model.save(model_filename)
    print(f"Model saved as: {model_filename}")
    
    # Save training history
    import json
    history_filename = f"resnet_history_{resnet_type}.json"
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'final_validation_accuracy': float(test_accuracy),
        'final_validation_loss': float(test_loss),
        'resnet_type': resnet_type,
        'epochs': len(history.history['accuracy'])
    }
    
    with open(history_filename, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved as: {history_filename}")
    
    # Testar com alguns exemplos
    print("\nTesting with sample data:")
    sample_images = X_val[:5]
    sample_labels = y_val[:5]
    
    predictions = model.predict(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)
    
    for i in range(len(sample_images)):
        print(f"Sample {i+1}:")
        print(f"  True label: {sample_labels[i]} ({classes[sample_labels[i]]})")
        print(f"  Predicted: {predicted_classes[i]} ({classes[predicted_classes[i]]})")
        print(f"  Confidence: {np.max(predictions[i]):.4f}")
        print()
