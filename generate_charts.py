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

    # Calculate R^2 (coefficient of determination)
    y_pred = polynomial(np.array(x))
    ss_res = np.sum((np.array(y) - y_pred) ** 2)
    ss_tot = np.sum((np.array(y) - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    if 'label' in kwargs:
        label = f'{kwargs.get("label")} (R^2={r_squared:.4f})'
        del kwargs['label']
        ax.plot(x_line, y_line, label=label, **kwargs)
    else:
        ax.plot(x_line, y_line, label=f'R^2={r_squared:.4f}', **kwargs)

    return r_squared


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
    r2_nn = polyfit(ax, nn_customers, nn_distances)
    print(f'NN R^2: {r2_nn:.4f}')
    ax.legend()
    fig.savefig('data/img/nn.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('SA')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(sa_customers, sa_distances)
    r2_sa = polyfit(ax, sa_customers, sa_distances)
    print(f'SA R^2: {r2_sa:.4f}')
    ax.legend()
    fig.savefig('data/img/sa.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('ACO')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(aco_customers, aco_distances)
    r2_aco = polyfit(ax, aco_customers, aco_distances)
    print(f'ACO R^2: {r2_aco:.4f}')
    ax.legend()
    fig.savefig('data/img/aco.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('GA')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(ga_customers, ga_distances)
    r2_ga = polyfit(ax, ga_customers, ga_distances)
    print(f'GA R^2: {r2_ga:.4f}')
    ax.legend()
    fig.savefig('data/img/ga.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('RL')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    ax.scatter(rl_customers, rl_distances)
    r2_rl = polyfit(ax, rl_customers, rl_distances)
    print(f'RL R^2: {r2_rl:.4f}')
    ax.legend()
    fig.savefig('data/img/rl.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 4.5)
    ax.set_title('Albany, NY')
    ax.set_xlabel('Customers')
    ax.set_ylabel('Total distance traveled (km)')
    r2_nn_comp  = polyfit(ax, nn_customers, nn_distances, color='blue', label='NN')
    r2_sa_comp  = polyfit(ax, sa_customers, sa_distances, color='red', label='SA')
    r2_aco_comp = polyfit(ax, aco_customers, aco_distances, color='orange', label='ACO')
    r2_ga_comp  = polyfit(ax, ga_customers, ga_distances, color='purple', label='GA')
    r2_rl_comp  = polyfit(ax, rl_customers, rl_distances, color='green', label='RL')
    print(f'Comparison - NN R^2: {r2_nn_comp:.4f}, SA R^2: {r2_sa_comp:.4f}, ACO R^2: {r2_aco_comp:.4f}, GA R^2: {r2_ga_comp:.4f}, RL R^2: {r2_rl_comp:.4f}')
    ax.legend()
    fig.savefig('data/img/comparison.png')


if __name__ == '__main__':
    main()
