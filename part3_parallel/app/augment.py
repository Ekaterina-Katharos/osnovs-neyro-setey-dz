import numpy as np

def apply_augmentation(image_array):
    """
    Применяет случайную аугментацию к изображению (поворот или добавление шума).
    """
    choice = np.random.choice(['flip', 'noise', 'none'])
    
    if choice == 'flip':
        # Отражение по горизонтали
        return np.fliplr(image_array)
    elif choice == 'noise':
        # Добавление случайного шума
        noise = np.random.normal(0, 0.1, image_array.shape)
        noisy_image = image_array + noise
        # Обрезаем значения, чтобы они были от 0 до 1
        return np.clip(noisy_image, 0, 1)
    else:
        return image_array