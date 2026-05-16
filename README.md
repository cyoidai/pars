# Package assignment and routing system (PARS)

Project whose aim is to investigate the capacitated vehicle routing problem (CVRP) using real-world road network information. CVRP is a type of vehicle routing problem (VRP), which itself is a generalization of the well-known traveling salesman problem (TSP).

It asks, given a set of vehicles each with a fixed capacity, and a set of customers, devise a route for each vehicle such that all customers are visited and we minimize the total distance traveled by all vehicles.

Our aim is to investigate efficient algorithms for solving the CVRP, notably classical heuristics such as nearest neighbor and sweep algorithm, along with metaheuristic solutions such as simulated annealing.

## Execution instructions

Install necessary packages

```bash
pip install stable-baselines3 gymnasium pyyaml networkx matplotlib numpy tensorboard osmnx sb3_contrib
```

The following command executes PARS on a 20km^2 map of Albany, NY centered at University of Albany, SUNY using nearest neighbor heuristic. Default parameters are 32 customers, 4 trucks, and a truck capacity of 8.

```bash
python main.py -a nn '1400 Washington Ave, Albany, NY 12222'
```

An example output is shown below. Take note of the final number, this number is the total distance traveled by all trucks in kilometers.

```text
('address', 'distance', 'customers', 'trucks', 'truck_capacity', 'algorithm', 'total_distance')
('1400 Washington Ave, Albany, NY 12222', 10000, 32, 4, 8, 'nn', 288.44087676199723)
```

For more information on arguments, run `python main.py -h`. The output is shown below.

```txt
usage: pars [-h] [--algorithm {nn,sa,aco,ga,rl}] [--distance DISTANCE] [--trucks TRUCKS] [--capacity CAPACITY] [--customers CUSTOMERS]
            [--draw-route DRAW_ROUTE] [--write-route WRITE_ROUTE] [--write-stats WRITE_STATS] [--iterations ITERATIONS]
            address

Package assignment and routing system (PARS)

positional arguments:
  address               Reference address

options:
  -h, --help            show this help message and exit
  --algorithm, -a {nn,sa,aco,ga,rl}
                        TSP optimizer
  --distance, -d DISTANCE
                        Distance
  --trucks, -t TRUCKS   Number of trucks
  --capacity, -c CAPACITY
                        Capacity of each truck
  --customers, -C CUSTOMERS
                        Total customers
  --draw-route DRAW_ROUTE
                        Draw the output route to the screen
  --write-route WRITE_ROUTE
                        Draw the output route to a file
  --write-stats WRITE_STATS
                        Write route statistics to a file
  --iterations, -i ITERATIONS
                        Number of iterations to perform
```
