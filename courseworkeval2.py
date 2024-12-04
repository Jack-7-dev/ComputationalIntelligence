import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from deap import creator, base, tools, algorithms
import matplotlib.pyplot as plt
import multiprocessing

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

    # Initialize counters for evaluations and accuracies
    evaluations_gd = []
    accuracies_gd = []
    evaluations_ga = []
    accuracies_ga = []
    evaluations_pso = []
    accuracies_pso = []
    total_evaluations = 100

    ### Gradient Descent Method
    def train_gradient_descent(model, trainloader, testloader): 
        print("Starting network training...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        num_evaluations = 0
        evaluations_gd = []
        accuracies_gd = []

        epochs = 20
        #batches_per_epoch = len(trainloader)
        #max_epochs = total_evaluations // batches_per_epoch
        print(f"Total epochs: {epochs}")
        
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

                num_evaluations += inputs.size(0)
                
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
            
            accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}, Accuracy: {accuracy:.2f}%')
            
            # Record evaluations and accuracy
            evaluations_gd.append(num_evaluations)
            accuracies_gd.append(accuracy)
        
        return evaluations_gd, accuracies_gd

    ### Genetic Algorithm
    def run_genetic_algorithm(model, testloader, generations=10, population_size=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print("Running Genetic Algorithm...")
        num_evaluations = 0
        generations = total_evaluations // population_size
        print(f"Total generations: {generations}")

        # Define fitness function
        def evaluate_fc_layer(individual):
            nonlocal num_evaluations
            num_evaluations += 1
            return evaluate_fc_layer_accuracy(individual, model, testloader)

        # Setup GA
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        num_weights = model.fc.weight.numel()
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_weights)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate_fc_layer)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Run GA
        population = toolbox.population(n=population_size)
        for gen in range(generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=population_size)

            # Record best accuracy
            best_individual = tools.selBest(population, k=1)[0]
            accuracy = evaluate_fc_layer_accuracy(best_individual, model, testloader)[0] * 100
            evaluations_ga.append(num_evaluations)
            accuracies_ga.append(accuracy)
            print(f"Generation {gen+1}, Best Accuracy: {accuracy:.2f}%, Evaluations: {num_evaluations}")

    ### Particle Swarm Optimization
    class ParticleSwarmOptimizer:
        def __init__(self, model, testloader, num_particles=10, iterations=10):
            self.model = model
            self.testloader = testloader
            self.num_particles = num_particles
            self.iterations = total_evaluations // num_particles
            self.num_evaluations = 0
            self.dim = model.fc.weight.numel()
            self.bounds = (-1, 1)

            # Initialize particles
            self.swarm = [self.Particle(self.dim, self.bounds) for _ in range(self.num_particles)]
            self.global_best_position = None
            self.global_best_score = float('-inf')

        class Particle:
            def __init__(self, dim, bounds):
                self.position = np.random.uniform(bounds[0], bounds[1], dim)
                self.velocity = np.random.uniform(-1, 1, dim)
                self.best_position = self.position.copy()
                self.best_score = float('-inf')

        def evaluate_fitness(self, weights):
            self.num_evaluations += 1
            return evaluate_fc_layer_accuracy(weights, self.model, self.testloader)[0]

        def optimize(self):
            print("Running PSO...")
            w = 0.7  # inertia
            c1 = 1.4  # cognitive parameter
            c2 = 1.4  # social parameter

            for iter in range(self.iterations):
                for particle in self.swarm:
                    fitness = self.evaluate_fitness(particle.position)
                    if fitness > particle.best_score:
                        particle.best_score = fitness
                        particle.best_position = particle.position.copy()
                    if fitness > self.global_best_score:
                        self.global_best_score = fitness
                        self.global_best_position = particle.position.copy()

                for particle in self.swarm:
                    inertia = w * particle.velocity
                    cognitive = c1 * np.random.rand(self.dim) * (particle.best_position - particle.position)
                    social = c2 * np.random.rand(self.dim) * (self.global_best_position - particle.position)
                    particle.velocity = inertia + cognitive + social
                    particle.position += particle.velocity

                # Record best accuracy
                accuracy = self.global_best_score * 100
                evaluations_pso.append(self.num_evaluations)
                accuracies_pso.append(accuracy)
                print(f"Iteration {iter+1}, Best Accuracy: {accuracy:.2f}%, Evaluations: {self.num_evaluations}")

    ### Helper Functions
    def evaluate_model(model, testloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
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
        accuracy = 100 * correct / total
        return accuracy

    def evaluate_fc_layer_accuracy(individual, model, testloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        # Set weights of fc layer from individual
        weights = torch.tensor(individual).float().reshape(model.fc.weight.shape).to(device)
        model.fc.weight.data = weights
        model.fc.bias.data.zero_()

        # Evaluate model
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
        accuracy = correct / total
        return (accuracy,)

    ### Run the methods
    # Clone the original model for each method to ensure fair comparison
    model_gd = CIFAR10ResNet()
    evaluations_gd, accuracies_gd = train_gradient_descent(model_gd, trainloader, testloader)

    model_ga = CIFAR10ResNet()
    run_genetic_algorithm(model_ga, testloader)

    model_pso = CIFAR10ResNet()
    pso_optimizer = ParticleSwarmOptimizer(model_pso, testloader)
    pso_optimizer.optimize()

    ### Plotting the results
    plt.plot(evaluations_gd, accuracies_gd, label='Gradient Descent')
    plt.plot(evaluations_ga, accuracies_ga, label='Genetic Algorithm')
    plt.plot(evaluations_pso, accuracies_pso, label='PSO')
    plt.xlabel('Number of Objective Function Evaluations')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Number of Function Evaluations')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()