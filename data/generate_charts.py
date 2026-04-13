import csv
import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        num_customers = []
        total_distances = []
        for i, (address, distance, customers, trucks, truck_capacity, total_distance) in enumerate(reader):
            if i == 0:
                continue
            num_customers.append(int(customers))
            total_distances.append(float(total_distance))
    return num_customers, total_distances

def polyfit(ax, x, y, degree: int=2, **kwargs):
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = polynomial(x_line)
    ax.plot(x_line, y_line, **kwargs)


def main():
    astar_customers, astar_distances = read_data('data/sweep_astar_nn.tsv')
    annealing_customers, annealing_distances = read_data('data/sweep_annealing.tsv')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('Sweep + A* with NN heuristic')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(astar_customers, astar_distances)
    polyfit(ax, astar_customers, astar_distances)
    fig.savefig('data/sweep_astar_nn.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.set_title('Sweep + Annealing')
    ax.scatter(annealing_customers, annealing_distances)
    polyfit(ax, annealing_customers, annealing_distances)
    fig.savefig('data/sweep_annealing.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    polyfit(ax, astar_customers, astar_distances, color='blue', label='A* with NN')
    polyfit(ax, annealing_customers, annealing_distances, color='red', label='Annealing')
    ax.legend()
    fig.savefig('data/comparison.png')


if __name__ == '__main__':
    main()
