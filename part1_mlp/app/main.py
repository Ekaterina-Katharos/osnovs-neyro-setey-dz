from models import NeuralNetwork
from manager import DatasetManager
from exceptions import NeuralNetError
import matplotlib.pyplot as plt
import numpy as np

network = None
dataset = DatasetManager()


def main_menu():
    global network
    while True:
        print("\n--- МЕНЮ НЕЙРОСЕТИ ---")
        print("1. Создать сеть")
        print("2. Загрузить данные")
        print("3. Обучить")
        print("4. Предсказать")
        print("5. Веса (сохр/загр)")
        print("6. График")
        print("7. Выход")

        choice = input("Выбор (1-7): ").strip()

        try:
            if choice == '1':
                msg = "Слои (через запятую, напр. 2,4,1): "
                layers = [int(x) for x in input(msg).split(',')]
                act = input("Активация (sigmoid/relu): ").strip()
                network = NeuralNetwork(layers, activation=act.lower())
                print("Сеть создана!")
            elif choice == '2':
                path = input("Путь к CSV: ").strip()
                dataset.prepare_data(path)
            elif choice == '3':
                if not network or dataset.X_train is None:
                    print("Нет данных для обучения!")
                    continue
                lr = float(input("Скорость обучения: "))
                epochs = int(input("Эпохи: "))
                batch = int(input("Размер батча: "))
                print("Обучение...")
                network.train(
                    dataset.X_train,
                    dataset.y_train,
                    epochs,
                    lr,
                    batch
                )
                print("Обучение завершено!")
            elif choice == '4':
                if not network or not network.is_trained:
                    print("Сеть не обучена!")
                    continue
                vals = [float(x) for x in input("Признаки: ").split()]
                scaled_vals = dataset.scaler.transform(np.array([vals]))
                res = network.predict(scaled_vals)[0][0]
                print(f"Предсказание: {res}")
            elif choice == '5':
                if not network:
                    print("Нет сети.")
                    continue
                sub = input("1.Сохранить 2.Загрузить: ")
                path = input("Имя файла (напр. weights.npz): ")
                if sub == '1':
                    network.save_weights(path)
                    print("Сохранено!")
                elif sub == '2':
                    network.load_weights(path)
                    print("Загружено!")
            elif choice == '6':
                if not network or not network.loss_history:
                    print("Нет данных.")
                    continue
                plt.plot(network.loss_history)
                plt.show()
            elif choice == '7':
                break
        except ValueError:
            print("Ошибка ввода числа!")
        except NeuralNetError as e:
            print(f"Ошибка сети: {e}")
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    main_menu()
