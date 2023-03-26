import matplotlib.pyplot as plt
import random
import time

from aco import AntColony


plt.style.use("dark_background")


COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    # (55, 20),
    # (50, 70),
    # (8, 70),
    # (95, 90),
    # (25, 60),
    # #(40, 40),
    # #(60, 70),
    # #(80, 40),
)


def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


start = time.time()
plot_nodes()

colony = AntColony(
    COORDS,
    ant_count=300,
    alpha=0.5,
    beta=1.2,
    pheromone_evaporation_rate=0.40,
    pheromone_constant=1000.0,
    iterations=300,
)

optimal_nodes = colony.get_path()
end = time.time()
for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

print(f"Time: {end - start}")
plt.show()

# Czas dla domyslnych wartosci i 7 wierzcholkow: 24.273500442504883 s
# Czas dla domyslnych wartosci i 10 wierzcholkow: 32.82139301300049 s
# Czas dla domyslnych wartosci i 15 wierzcholkow: 54.09216499328613 s


# Zmienione wartosci oraz domyslne wierzcholki:
# ant-count
# (COORDS,ant_count=500,alpha=0.5,beta=1.2,pheromone_evaporation_rate=0.40,pheromone_constant=1000.0,iterations=300): 44.86803960800171 s
# (COORDS,ant_count=100,alpha=0.5,beta=1.2,pheromone_evaporation_rate=0.40,pheromone_constant=1000.0,iterations=300): 10.44399881362915 s
# alpha
# (COORDS,ant_count=300,alpha=0.8,beta=1.2,pheromone_evaporation_rate=0.40,pheromone_constant=1000.0,iterations=300): 26.153483152389526 s
# (COORDS,ant_count=300,alpha=0.2,beta=1.2,pheromone_evaporation_rate=0.40,pheromone_constant=1000.0,iterations=300): 30.40099859237671 s
# beta
# (COORDS,ant_count=300,alpha=0.5,beta=1.6,pheromone_evaporation_rate=0.40,pheromone_constant=1000.0,iterations=300): 29.592498540878296 s
# (COORDS,ant_count=300,alpha=0.5,beta=0.8,pheromone_evaporation_rate=0.40,pheromone_constant=1000.0,iterations=300): 27.331499814987183 s
# pheromone_evaporation_rate
# (COORDS,ant_count=300,alpha=0.5,beta=1.2,pheromone_evaporation_rate=0.60,pheromone_constant=1000.0,iterations=300): 26.301499843597412 s
# (COORDS,ant_count=300,alpha=0.5,beta=1.2,pheromone_evaporation_rate=0.20,pheromone_constant=1000.0,iterations=300): 25.015501737594604 s
# pheromone_constant
# (COORDS,ant_count=300,alpha=0.5,beta=1.2,pheromone_evaporation_rate=0.40,pheromone_constant=2000.0,iterations=300): 25.103498935699463 s
# (COORDS,ant_count=300,alpha=0.5,beta=1.2,pheromone_evaporation_rate=0.40,pheromone_constant=500.0,iterations=300): 24.125999927520752 s

# Ilość wierzchołków wpływa na czas wykonania znacząco, ponieważ zwiększa się ilość możliwych ścieżek, które muszą być sprawdzone. (Im wiecej wierzholkow tym wiekszy czas wykonania)
# Ilość mrówek wpływa na czas wykonania znacząco (Im wiecej mrówek tym wiekszy czas wykonania)
# Zmiana wartości alpha na niższą oraz wyższą spodowała niewielki wzrost czasu wykonania.
# Zmiana wartości beta na niższą oraz wyższą spodowała niewielki wzrost czasu wykonania, z czego zmiana na niższą spowodowała wiekszy wzrost
# Zmiana wartości pheromone_evaporation_rate na niższą oraz wyższą spodowała niewielki wzrost czasu wykonania, z czego zmiana na niższą spowodowała minimalnie wiekszy wzrost
# Zmiana wartości pheromone_constant na niższą spowodowała niewielki wzrost czasu wykonania, a na wyższą czas pozostał bez zmian.
# Liczba iteracji oczywiście znacząco wpływa na czas wykonania, im więcej tym dłużej trwa sprawdzanie wszystkich możliwych ścieżek.
