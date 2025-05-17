import csv
import numpy as np
import matplotlib.pyplot as plt


def read_csv_file(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for row in reader:
            data.append([float(val) for val in row])
    return headers, np.array(data)


def calculate_statistics(data):
    stats = {
        'count': len(data),
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data)
    }
    return stats


def linear_regression(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    b = (sum_y - a * sum_x) / n

    return a, b


def plot_data(ax, x, y, x_label, y_label, title):
    ax.scatter(x, y, color='blue', label='Исходные данные')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def plot_regression_line(ax, x, y, a, b, x_label, y_label):
    ax.scatter(x, y, color='blue', label='Исходные данные')
    ax.plot(x, a * x + b, color='red', label=f'Регрессия: y = {a:.2f}x + {b:.2f}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Регрессионная прямая')
    ax.grid(True)
    ax.legend()


def plot_error_squares(ax, x, y, a, b, x_label, y_label):
    ax.scatter(x, y, color='blue', label='Исходные данные')
    ax.plot(x, a * x + b, color='red', label=f'Регрессия: y = {a:.2f}x + {b:.2f}')

    for xi, yi in zip(x, y):
        y_pred = a * xi + b
        ax.plot([xi, xi], [yi, y_pred],
                color='green', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Ошибки регрессии')
    ax.grid(True)
    ax.legend()


def main():

    filename = 'student_scores.csv'
    headers, data = read_csv_file(filename)

    print(f"Доступные столбцы: 0 - {headers[0]}, 1 - {headers[1]}")
    x_col = int(input("Выберите столбец для X (0 или 1): "))
    y_col = 1 - x_col if x_col in (0, 1) else 0

    x = data[:, x_col]
    y = data[:, y_col]
    x_label = headers[x_col]
    y_label = headers[y_col]

    x_stats = calculate_statistics(x)
    y_stats = calculate_statistics(y)

    print("\nСтатистика по столбцу X:")
    print(f"Количество: {x_stats['count']}")
    print(f"Минимум: {x_stats['min']:.2f}")
    print(f"Максимум: {x_stats['max']:.2f}")
    print(f"Среднее: {x_stats['mean']:.2f}")

    print("\nСтатистика по столбцу Y:")
    print(f"Количество: {y_stats['count']}")
    print(f"Минимум: {y_stats['min']:.2f}")
    print(f"Максимум: {y_stats['max']:.2f}")
    print(f"Среднее: {y_stats['mean']:.2f}")

    a, b = linear_regression(x, y)
    print(f"\nПараметры регрессионной прямой: a = {a:.4f}, b = {b:.4f}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    plot_data(ax1, x, y, x_label, y_label, 'Исходные данные')
    plot_regression_line(ax2, x, y, a, b, x_label, y_label)
    plot_error_squares(ax3, x, y, a, b, x_label, y_label)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()