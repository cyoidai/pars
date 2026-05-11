import csv
import matplotlib.pyplot as plt
import numpy as np


def read_data(path, algorithm: str):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        num_customers = []
        total_distances = []
        for i, (address, distance, customers, trucks, truck_capacity, algo, total_distance) in enumerate(reader):
            if i == 0:
                continue
            if algo == algorithm:
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

    nn_customers, nn_distances   = read_data('output.tsv', 'nn')
    sa_customers, sa_distances   = read_data('output.tsv', 'sa')
    aco_customers, aco_distances = read_data('output.tsv', 'aco')
    ga_customers, ga_distances   = read_data('output.tsv', 'ga')
    rl_customers, rl_distances   = read_data('output.tsv', 'rl')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('NN')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(nn_customers, nn_distances)
    polyfit(ax, nn_customers, nn_distances)
    fig.savefig('data/img/nn.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('SA')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(sa_customers, sa_distances)
    polyfit(ax, sa_customers, sa_distances)
    fig.savefig('data/img/sa.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('ACO')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(aco_customers, aco_distances)
    polyfit(ax, aco_customers, aco_distances)
    fig.savefig('data/img/aco.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('GA')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(ga_customers, ga_distances)
    polyfit(ax, ga_customers, ga_distances)
    fig.savefig('data/img/ga.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('RL')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(rl_customers, rl_distances)
    polyfit(ax, rl_customers, rl_distances)
    fig.savefig('data/img/rl.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('Albany, NY')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    polyfit(ax, nn_customers, nn_distances, color='blue', label='NN')
    polyfit(ax, sa_customers, sa_distances, color='red', label='SA')
    polyfit(ax, aco_customers, aco_distances, color='orange', label='ACO')
    polyfit(ax, ga_customers, ga_distances, color='purple', label='GA')
    polyfit(ax, rl_customers, rl_distances, color='green', label='RL')
    ax.legend()
    fig.savefig('data/img/comparison.png')


if __name__ == '__main__':
    main()
