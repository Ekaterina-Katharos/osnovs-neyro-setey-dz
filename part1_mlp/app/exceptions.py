class NeuralNetError(Exception):
    """Базовый класс для исключений нейросети."""

    pass


class InvalidLayerSizeError(NeuralNetError):
    """Вызывается при неверном размере слоев."""

    pass


class MismatchedDataError(NeuralNetError):
    """Вызывается при несовпадении размерности данных."""

    pass


class ModelNotTrainedError(NeuralNetError):
    """Вызывается при попытке предсказания необученной сетью."""

    pass
