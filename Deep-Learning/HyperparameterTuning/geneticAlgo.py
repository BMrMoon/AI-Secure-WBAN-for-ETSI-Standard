from icecream import ic
from tqdm import tqdm
import random
from deap import base, creator, tools, algorithms
from Models.models import FC_Model, CNN1d, RNN, GRU, LSTM
import torch
from prettytable import PrettyTable
import gc

ic(f"PyTorch version: {torch.__version__}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
ic(f"Using device: {device}")


MODEL_CLASSES = {
    "FC_Model": FC_Model,
    "CNN1d": CNN1d,
    "RNN": RNN,
    "GRU": GRU,
    "LSTM": LSTM
}
"""MODEL_CLASSES = {
    "ELM": ELM
}"""

creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

class GeneticAlgo:
    def __init__(self, dataset, param_ranges):
        self.dataset = dataset
        self.param_ranges = param_ranges
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.toolbox = base.Toolbox()

    def register_model_toolbox(self, model_name):
        self.toolbox = base.Toolbox()
        param_ranges = self.param_ranges[model_name]

        attr_funcs = []
        self.param_keys = list(param_ranges.keys())

        # Register attribute generators
        for key, val in param_ranges.items():
            if isinstance(val, tuple):
                if isinstance(val[0], int):
                    self.toolbox.register(key, random.randint, val[0], val[1])
                else:
                    self.toolbox.register(key, random.uniform, val[0], val[1])
            elif isinstance(val, list):
                self.toolbox.register(key, random.choice, val)
            else:
                raise ValueError(f"Invalid range type for {key}: {val}")
            attr_funcs.append(getattr(self.toolbox, key))

        # Individual & population
        self.toolbox.register("individual", tools.initCycle, creator.Individual, attr_funcs, n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Type-aware custom mutation
        def custom_mutate(individual, indpb=0.2):
            for i, key in enumerate(self.param_keys):
                if random.random() < indpb:
                    val = param_ranges[key]
                    if isinstance(val, tuple):
                        if isinstance(val[0], int):
                            individual[i] += random.choice([-1, 1])
                            individual[i] = max(val[0], min(val[1], individual[i]))
                        else:  # float
                            individual[i] += random.gauss(0, 0.1 * (val[1] - val[0]))
                            individual[i] = max(val[0], min(val[1], individual[i]))
                    elif isinstance(val, list):
                        individual[i] = random.choice(val)
            return individual,

        # Type-aware custom crossover
        def custom_crossover(ind1, ind2):
            for i, key in enumerate(self.param_keys):
                if random.random() < 0.5:
                    ind1[i], ind2[i] = ind2[i], ind1[i]
            return ind1, ind2

        # Register genetic operators
        self.toolbox.register("mate", custom_crossover)
        self.toolbox.register("mutate", custom_mutate)
        self.toolbox.register("select", tools.selNSGA2)

    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def evaluate_model(self, individual, model_name):
        #ic(individual)
        param_keys = self.param_keys
        hyperparameters = dict(zip(param_keys, individual))
        #ic(hyperparameters)
        

        configurations = {
            "device": self.device,
            "dataset": self.dataset
        }

        
        ModelClass = MODEL_CLASSES[model_name]
        model = ModelClass(hyperparameters, configurations, self.param_ranges).to(self.device)
        #total_params = self.count_parameters(model)

        try:
            loss, acc, val_loss, val_acc, Accuracies, Losses = model.run(model)
            del model
            torch.mps.empty_cache()
            gc.collect()
            #penalty = max(0, eval_time - 0.01) * 10
            return loss, acc, val_loss, val_acc
        except Exception as e:
            print(f"[{model_name}] Evaluation failed: {e}")
            return 9999.0, 0.0, 9999.0, 0.0
        

    def modelGA(self, model_name, pop_size, ngen):
        print(f"\n🚀 Optimizing {model_name}")
        self.register_model_toolbox(model_name)
        self.toolbox.register("evaluate", lambda ind: self.evaluate_model(ind, model_name))

        pop = self.toolbox.population(n=pop_size)

        # 🌟 ADDED: Evaluate initial population
        for ind in pop:
            if not ind.fitness.valid:
                try:
                    eval_result = self.toolbox.evaluate(ind)
                    if eval_result is None or not isinstance(eval_result, tuple):
                        eval_result = (9999.0, 0.0, 9999.0, 0.0)
                    ind.fitness.values = eval_result
                except Exception as e:
                    print(f"⚠️ Initial evaluation exception for individual {ind}: {e}")
                    ind.fitness.values = (9999.0, 0.0, 9999.0, 0.0)

        for gen in range(ngen):
            print(f"📈 Generation {gen+1}/{ngen}")
            offspring = algorithms.varAnd(pop, self.toolbox, cxpb=0.5, mutpb=0.2)
            for ind in offspring:
                try:
                    eval_result = self.toolbox.evaluate(ind)
                    if eval_result is None or not isinstance(eval_result, tuple):
                        eval_result = (9999.0, 0.0, 9999.0, 0.0)
                    ind.fitness.values = eval_result
                except Exception as e:
                    print(f"⚠️ Evaluation exception for individual {ind}: {e}")
                    ind.fitness.values = (9999.0, 0.0, 9999.0, 0.0)
            pop = self.toolbox.select(pop + offspring, k=len(pop))

        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

        print("\n🏆 Best individuals (Pareto front):")
        for ind in pareto_front:
            # 🌟 ADDED: Skip invalid individuals
            if not hasattr(ind.fitness, 'values') or len(ind.fitness.values) < 4:
                print(f"⚠️ Skipping invalid individual: {ind}")
                continue
            print(f"Model: {model_name}, Params: {dict(zip(self.param_keys, ind))}, "
                f"Loss: {ind.fitness.values[0]:.4f}, Acc: {ind.fitness.values[1]:.4f}, "
                f"Validation Loss: {ind.fitness.values[2]:.4f}, Validation Acc: {ind.fitness.values[3]:.4f}")

        return pareto_front
    
    def run_all_models(self, pop_size, ngen):
        best_overall = None
        best_model_name = None
        best_fitness = (float("inf"), -float("inf"), float("inf"), -float("inf"))  # (loss, acc, val_loss, val_acc)
        w_loss = 0.375
        w_acc = 0.125
        w_val_loss = 0.375
        w_val_acc = 0.125
        best_score = float("inf")

        # NEW: Dictionary to store best individual for each model class
        best_per_model = {}

        for model_name in self.param_ranges.keys():
            pareto_front = self.modelGA(model_name, pop_size, ngen)

            best_local = None
            best_local_score = float("inf")

            for ind in pareto_front:
                loss, acc, val_loss, val_acc = ind.fitness.values
                score = (w_loss * loss) - (w_acc * acc) + (w_val_loss * val_loss) - (w_val_acc * val_acc)

                # Track best in this model
                if score < best_local_score:
                    best_local_score = score
                    best_local = {
                        "model": model_name,
                        "params": dict(zip(self.param_keys, ind)),
                        "loss": loss,
                        "accuracy": acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc
                    }

                # Track best overall
                if score < best_score:
                    best_score = score
                    best_overall = {
                        "model": model_name,
                        "params": dict(zip(self.param_keys, ind)),
                        "loss": loss,
                        "accuracy": acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc
                    }

            best_per_model[model_name] = best_local

        # 🎯 Show Best Per Model
        print("\n📊 Best Individual from Each Model:")
        for model_name, best_ind in best_per_model.items():
            print(f"🔹 Model: {model_name}")
            print(f"    Params: {best_ind['params']}")
            print(f"    Loss: {best_ind['loss']:.4f}")
            print(f"    Accuracy: {best_ind['accuracy']:.4f}")
            print(f"    Validation Loss: {best_ind['val_loss']:.4f}")
            print(f"    Validation Accuracy: {best_ind['val_accuracy']:.4f}")

        # 🏅 Show Best Overall
        print("\n🏅 Best of All Models:")
        print(f"Model: {best_overall['model']}")
        print(f"Params: {best_overall['params']}")
        print(f"Loss: {best_overall['loss']:.4f}")
        print(f"Accuracy: {best_overall['accuracy']:.4f}")
        print(f"Validation Loss: {best_overall['val_loss']:.4f}")
        print(f"Validation Accuracy: {best_overall['val_accuracy']:.4f}")

        return best_overall, best_per_model

