import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def f(x,y):
    return np.sin(x)*np.exp(pow(1-np.cos(y), 2)) + np.cos(y)*np.exp(pow(1-np.sin(x), 2)) + pow((x-y),2)

def plot_3d(points):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.view_init(elev=100.)
    
    # Plot graph
    ax.contour(X, Y, Z, levels=8, cmap='summer', zorder=0.5)
    ax.set_title('bird')

    # Plot points
    xdata = np.array(points).T[0]
    ydata = np.array(points).T[1]
    zdata = f(xdata, ydata)
    ax.scatter(xdata, ydata, zdata, c='black', linewidths=1, zorder=1)
    
    fig.show()

def fitness(data):
    df = pd.DataFrame()
    df["x"] = np.array(data).T[0]
    df["y"] = np.array(data).T[1]
    df["fitness"] = f(np.array(data).T[0], np.array(data).T[1])
    return df

def clone_mutate_and_select(df: pd.DataFrame, clone_n, p):

    sorted_df = df.sort_values(by=['fitness'], ascending=True, ignore_index=True)

    new_population = []

    # Set fitness and ranking variables
    Dmax = 100
    sorted_df['ranking'] = -1
    sorted_df['ranking'][0] = 100
    sorted_df['ranking'][len(sorted_df) - 1] = 0
    for i in sorted_df.index:
        if sorted_df['ranking'][i] == -1:
            sorted_df['ranking'][i] = 100*((len(sorted_df) - i - 1)/(len(sorted_df) - 1))

    for i, individual in sorted_df.iterrows():

        D = individual['ranking']/Dmax
        alpha = np.exp(-p*D)

        local_df = pd.DataFrame()
        local_df['x'] = [0.0 for x in range(0, clone_n)]
        local_df['y'] = [0.0 for x in range(0, clone_n)]
        local_df['fitness'] = [0.0 for x in range(0, clone_n)]

        for it in range(0, clone_n):

            # Checks if mutation will happen
            if random.uniform(0,1) > alpha:
                local_df['x'][it] = individual['x']
                local_df['y'][it] = individual['y']
                local_df['fitness'][it] = individual['fitness']
                continue

            # Generate new random values for the mutated individual
            local_df['x'][it] = random.uniform(-10, 10)
            local_df['y'][it] = random.uniform(-10, 10)
            local_df['fitness'][it] = f(local_df['x'][it],local_df['y'][it])

        # Select best local individual
        local_df = local_df.sort_values(by=['fitness'], ascending=True, ignore_index=True)
        new_population.append([local_df['x'][0], local_df['y'][0]])
    
    return new_population

if __name__ == "__main__":

    size = 100
    generations = 50
    clone_n = 5
    p = 5
    
    population = []

    for x in range(0,size):
        population.append([random.uniform(-10, 10),random.uniform(-10, 10)])

    for gen in range(0,generations):

        population_df = fitness(population)

        population = clone_mutate_and_select(population_df, clone_n, p)

    plot_3d(population)
    
    print('test')