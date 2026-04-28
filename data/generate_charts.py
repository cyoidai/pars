import csv
import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    customers = []
    distances = []

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')

        for i, row in enumerate(reader):
            if i == 0:
                continue

            # supports both old and new format
            if len(row) == 6:
                address, distance, num_customers, trucks, truck_capacity, total_distance = row
            else:
                algorithm, address, distance, num_customers, trucks, truck_capacity, total_distance = row

            customers.append(int(num_customers))
            distances.append(float(total_distance))

    return customers, distances


def polyfit(ax, x, y, label=None):
    if len(x) < 3:
        ax.plot(x, y, marker='o', label=label)
        return

    coefficients = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coefficients)

    x_line = np.linspace(min(x), max(x), 100)
    y_line = polynomial(x_line)

    ax.scatter(x, y)
    ax.plot(x_line, y_line, label=label)


def main():
    sa_x, sa_y = read_data('sweep_sa.tsv')
    ga_x, ga_y = read_data('sweep_ga.tsv')
    aco_x, aco_y = read_data('sweep_aco.tsv')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)

    ax.set_title('Algorithm Comparison: SA vs GA vs ACO')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total Distance (km)')

    polyfit(ax, sa_x, sa_y, label='Simulated Annealing')
    polyfit(ax, ga_x, ga_y, label='Genetic Algorithm')
    polyfit(ax, aco_x, aco_y, label='Ant Colony Optimization')

    ax.legend()
    fig.savefig('algorithm_comparison.png')

    print("Saved: algorithm_comparison.png")


if __name__ == '__main__':
    main()