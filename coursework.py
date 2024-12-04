import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt

def main():
    print("Starting main function...")
    # Set up multiprocessing
    multiprocessing.freeze_support()

    # Data transformations
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # ResNet Block Definition
    class ResNetBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResNetBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

    # CIFAR-10 ResNet Model
    class CIFAR10ResNet(nn.Module):
        def __init__(self, num_classes=10):
            super(CIFAR10ResNet, self).__init__()
            self.prep = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
            
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(256, num_classes)

        def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
            layers = []
            layers.append(ResNetBlock(in_channels, out_channels, stride))
            for _ in range(1, num_blocks):
                layers.append(ResNetBlock(out_channels, out_channels))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.prep(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    # Training Function
    def train_network(model, trainloader, epochs=5):  # Reduced epochs to 5 for testing
        print("Starting network training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f"[{epoch+1}, {i+1}] loss: {running_loss / 100:.4f}")
                    running_loss = 0.0
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f'Epoch {epoch+1}, Accuracy: {100 * correct / total:.2f}%')
    
    # Initial full network training
    model = CIFAR10ResNet()
    train_network(model, trainloader)

    # def evaluate_fc_layer(individual, model, testloader):
    #     print("Evaluating fully connected layer...")
    #     # Convert individual to layer weights
    #     weights = torch.tensor(individual).float().reshape(model.fc.weight.shape)
        
    #     # Freeze all layers except the last
    #     for param in model.parameters():
    #         param.requires_grad = False
        
    #     # Only last layer is trainable
    #     model.fc.weight.data = weights
    #     model.fc.bias.data.zero_()
        
    #     # Evaluate performance
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = model.to(device)
    #     with torch.no_grad():
    #         for inputs, labels in testloader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
        
    #     accuracy = correct / total
    #     print(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    #     return (accuracy,)  # DEAP works with tuples

    # # Setup for genetic algorithm
    # print("Setting up genetic algorithm...")
    # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # creator.create("Individual", list, fitness=creator.FitnessMax)

    # toolbox = base.Toolbox()
    # toolbox.register("attr_float", np.random.uniform, -1, 1)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=model.fc.weight.numel())
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("evaluate", evaluate_fc_layer, model=model, testloader=testloader)
    # toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    # toolbox.register("select", tools.selTournament, tournsize=3)

    # def run_genetic_algorithm():
    #     print("Running genetic algorithm...")
    #     population = toolbox.population(n=50)
    #     algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
    #     best_individual = tools.selBest(population, k=1)[0]
    #     accuracy = evaluate_fc_layer(best_individual, model, testloader)[0]
    #     print(f"Genetic algorithm complete. Best accuracy: {accuracy:.4f}")
    #     return accuracy

    # class ParticleSwarmOptimizer:
    #     def __init__(self, model, testloader):
    #         print("Initializing Particle Swarm Optimizer...")
    #         self.model = model
    #         self.testloader = testloader
    #         self.n_particles = 30
    #         self.w = 0.7  # inertia
    #         self.c1 = 1.4  # cognitive parameter
    #         self.c2 = 1.4  # social parameter
    #         self.dim = model.fc.weight.numel()
    #         self.bounds = (-1, 1)
    #         self.swarm = [self.Particle(self.dim, self.bounds) for _ in range(self.n_particles)]

    #     class Particle:
    #         def __init__(self, dim, bounds):
    #             self.position = np.random.uniform(bounds[0], bounds[1], dim)
    #             self.velocity = np.random.uniform(-1, 1, dim)
    #             self.best_position = self.position.copy()
    #             self.best_score = float('-inf')

    #     def evaluate_fitness(self, weights):
    #         print("Evaluating fitness of a particle...")
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #         self.model = self.model.to(device)
    #         return evaluate_fc_layer(weights, self.model, self.testloader)[0]

    #     def optimize(self, iterations=40):
    #         print("Starting PSO optimization...")
    #         global_best_position = None
    #         global_best_score = float('-inf')

    #         for iteration in range(iterations):
    #             print(f"Iteration {iteration+1}/{iterations}")
    #             for particle in self.swarm:
    #                 fitness = self.evaluate_fitness(particle.position)
    #                 if fitness > particle.best_score:
    #                     particle.best_score = fitness
    #                     particle.best_position = particle.position.copy()
    #                 if fitness > global_best_score:
    #                     global_best_score = fitness
    #                     global_best_position = particle.position.copy()

    #             for particle in self.swarm:
    #                 inertia = self.w * particle.velocity
    #                 cognitive = self.c1 * np.random.rand(self.dim) * (particle.best_position - particle.position)
    #                 social = self.c2 * np.random.rand(self.dim) * (global_best_position - particle.position)
    #                 particle.velocity = inertia + cognitive + social
    #                 particle.position += particle.velocity

    #         print(f"PSO optimization complete. Best score: {global_best_score:.4f}")
    #         return global_best_score

    # # Commented out differential evolution
    # # def objective_function(weights):
    # #     print("Evaluating objective function...")
    # #     return -evaluate_fc_layer(weights, model, testloader)[0]

    # # def differential_evolution_callback(xk, convergence):
    # #     print(f"Generation {differential_evolution_callback.generation}: Best score so far: {-objective_function(xk):.4f}")
    # #     differential_evolution_callback.generation += 1

    # # differential_evolution_callback.generation = 1

    # # print("Running differential evolution...")
    # # result = differential_evolution(
    # #     objective_function,
    # #     bounds=[(-1, 1) for _ in range(model.fc.weight.numel())],
    # #     maxiter=3,  # Maximum number of generations
    # #     popsize=3,  # Population size
    # #     tol=0.01,   # Convergence tolerance
    # #     disp=True,  # Display progress
    # #     callback=differential_evolution_callback
    # # )
    # # differential_evolution_accuracy = -result.fun
    # # print(f"Differential evolution complete. Best accuracy: {differential_evolution_accuracy:.4f}")

    # def compare_optimization_methods(model, testloader):
    #     print("Comparing optimization methods...")
    #     methods = ['Standard Adam', 'Genetic Algorithm', 'PSO']
    #     accuracies = []

    #     # Standard Adam accuracy
    #     print("Evaluating Standard Adam accuracy...")
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = model.to(device)
    #     with torch.no_grad():
    #         for inputs, labels in testloader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #     adam_accuracy = 100 * correct / total
    #     accuracies.append(adam_accuracy)
    #     print(f"Standard Adam accuracy: {adam_accuracy:.2f}%")

    #     # Genetic Algorithm accuracy
    #     genetic_algorithm_accuracy = run_genetic_algorithm()
    #     accuracies.append(100 * genetic_algorithm_accuracy)

    #     # PSO accuracy
    #     pso = ParticleSwarmOptimizer(model, testloader)
    #     pso_accuracy = pso.optimize()
    #     accuracies.append(100 * pso_accuracy)

    #     # Commented out differential evolution accuracy
    #     # accuracies.append(100 * differential_evolution_accuracy)

    #     plt.bar(methods, accuracies)
    #     plt.title('Last Layer Optimization Comparison')
    #     plt.ylabel('Accuracy')
    #     plt.show()

    # compare_optimization_methods(model, testloader)

if __name__ == '__main__':
    main()