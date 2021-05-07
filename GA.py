from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import TournamentSelection, UniformCrossover, FlipBitMutation, ExponentialRankingSelection
from gaft.analysis import FitnessStore, ConsoleOutput
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from math import sin, sqrt
import matplotlib.pyplot as plt

def plot_best_fit():
    from best_fit import best_fit
    import re
    geracoes, inds, scores = list(zip(*best_fit))
    
    n_nines = []
    for s in scores:
        precisao = str(s).split('.')[1]
        noves = re.search(r'(^9*)', precisao)
        n_nines.append(len(noves[0]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(geracoes,n_nines)
    plt.show()

if __name__ == "__main__":

    individuo = BinaryIndividual(ranges=[(-100, 100),(-100, 100)], eps=0.000001)
    populacao = Population(indv_template=individuo, size=100)
    populacao.init()

    selecao = TournamentSelection()
    crossover = UniformCrossover(pc=0.65, pe=0.65)
    mutacao = FlipBitMutation(pm=0.008)

    engine = GAEngine(population=populacao, selection=selecao,
                        crossover=crossover, mutation=mutacao,
                        analysis=[FitnessStore, ConsoleOutput])
    
    @engine.fitness_register
    def aptidao(ind):
        x,y = ind.solution
        return  0.5 - ((sin(sqrt(x**2 + y**2))**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2)
    
    engine.run(ng=40)

    plot_best_fit()
