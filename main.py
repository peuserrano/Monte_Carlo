
from src.modelling import MonteCarlo

if __name__ == '__main__':

    # Exemplo de uso:
    model = MonteCarlo(['PETR4','WEGE3','BPAC11','VALE3'], 
                        simulations=1000, 
                        projected_days=252, 
                        starting_ammount=100000)

    model.pulling_stock_data()
    model.simulating_scenarios()
    model.plotting_scenarios()
    model.performance_evaluation()

    
    
