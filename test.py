for particle in self.swarm:
    fitness = self.evaluate_fitness(particle.position)
    if fitness > particle.best_score:
        particle.best_score = fitness
        particle.best_position = particle.position.copy()
    if fitness > self.global_best_score:
        self.global_best_score = fitness
        self.global_best_position = particle.position.copy()

for particle in self.swarm:
    fitness = self.evaluate_fitness(particle.position)
    if fitness > particle.best_score:  # Now comparing floats
        particle.best_score = fitness
        particle.best_position = particle.position.copy()
    if fitness > self.global_best_score:
        self.global_best_score = fitness
        self.global_best_position = particle.position.copy()