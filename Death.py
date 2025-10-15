import os
import sys
import tty
import termios
import random
import math
import numpy as np
from collections import deque, defaultdict
import time
import select
from enum import Enum

# =============================
# Enhanced Color System
# =============================

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    ORANGE = '\033[38;5;214m'
    DARK_RED = '\033[38;5;88m'
    DARK_GREEN = '\033[38;5;22m'
    
    @staticmethod
    def color_text(text, color):
        return f"{color}{text}{Colors.RESET}"

# =============================
# Advanced Neural Network
# =============================

class AdvancedNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layers = []
        self.activations = []
        
        # Input to first hidden layer
        self.layers.append(np.random.randn(hidden_sizes[0], input_size) * 0.1)
        self.activations.append(np.zeros(hidden_sizes[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(np.random.randn(hidden_sizes[i], hidden_sizes[i-1]) * 0.1)
            self.activations.append(np.zeros(hidden_sizes[i]))
        
        # Output layer
        self.layers.append(np.random.randn(output_size, hidden_sizes[-1]) * 0.1)
        self.activations.append(np.zeros(output_size))
    
    def forward(self, x):
        x = np.array(x).flatten()
        
        # Forward pass through layers
        current_activation = x
        for i, layer in enumerate(self.layers):
            current_activation = np.dot(layer, current_activation)
            if i < len(self.layers) - 1:  # Hidden layers use relu
                current_activation = np.maximum(0, current_activation)
            else:  # Output layer uses softmax
                current_activation = self._softmax(current_activation)
            
            self.activations[i] = current_activation
        
        return current_activation
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.3):
        for i in range(len(self.layers)):
            mutation_mask = np.random.random(self.layers[i].shape) < mutation_rate
            mutations = np.random.randn(*self.layers[i].shape) * mutation_strength
            self.layers[i] += mutation_mask * mutations
    
    def crossover(self, other):
        child = AdvancedNeuralNetwork(1, [self.layers[0].shape[0]], self.layers[-1].shape[0])
        
        for i in range(len(self.layers)):
            crossover_mask = np.random.random(self.layers[i].shape) > 0.5
            child.layers[i] = np.where(crossover_mask, self.layers[i], other.layers[i])
        
        return child

# =============================
# Advanced AI System
# =============================

class AIState(Enum):
    PATROL = "patrol"
    COMBAT = "combat"
    SEARCH = "search"
    RETREAT = "retreat"
    SUPPORT = "support"
    HEAL = "heal"
    FLANK = "flank"
    AMBUSH = "ambush"

class AISquadRole(Enum):
    LEADER = "leader"
    ASSAULT = "assault"
    SUPPORT = "support"
    MEDIC = "medic"
    SNIPER = "sniper"

class TacticalMemory:
    def __init__(self, max_memories=50):
        self.memories = deque(maxlen=max_memories)
        self.last_known_positions = deque(maxlen=10)
        self.player_pattern = deque(maxlen=15)
        self.communication_log = deque(maxlen=20)
    
    def add_memory(self, position, certainty, source, timestamp=None):
        memory = {
            'position': position,
            'certainty': certainty,
            'source': source,
            'timestamp': timestamp or time.time()
        }
        self.memories.append(memory)
        
        if certainty > 0.7:
            self.last_known_positions.append(position)
    
    def get_recent_memories(self, time_window=30):
        current_time = time.time()
        return [m for m in self.memories if current_time - m['timestamp'] < time_window]
    
    def predict_player_position(self, steps=1):
        if len(self.player_pattern) < 3:
            return None
        
        # Use linear regression for prediction
        positions = list(self.player_pattern)
        n = len(positions)
        
        if n < 2:
            return positions[-1] if positions else None
        
        # Simple linear extrapolation
        dx = positions[-1][0] - positions[-2][0]
        dy = positions[-1][1] - positions[-2][1]
        
        predicted_x = positions[-1][0] + dx * steps
        predicted_y = positions[-1][1] + dy * steps
        
        return (int(predicted_x), int(predicted_y))

class AdvancedTacticalAI:
    def __init__(self, genome_id=None):
        self.genome_id = genome_id or random.randint(1000, 9999)
        self.memory = TacticalMemory()
        self.state = AIState.PATROL
        self.squad_role = None
        self.squad_members = []
        self.assigned_target = None
        
        # Genetic traits
        self.genetic_traits = {
            'aggression': random.uniform(0.2, 0.9),
            'caution': random.uniform(0.1, 0.8),
            'teamwork': random.uniform(0.3, 0.95),
            'accuracy': random.uniform(0.6, 1.0),
            'mobility_preference': random.uniform(0.3, 0.9),
            'communication_frequency': random.uniform(0.2, 0.8)
        }
        
        # Neural network for decision making
        self.neural_net = AdvancedNeuralNetwork(
            input_size=24,
            hidden_sizes=[16, 12, 8],
            output_size=8
        )
        
        # Performance tracking
        self.performance = {
            'damage_dealt': 0,
            'damage_taken': 0,
            'kills': 0,
            'assists': 0,
            'survival_time': 0,
            'communication_count': 0
        }
        
        # Communication system
        self.communication_range = 12
        self.last_communication = 0
        self.known_threats = {}
        
        # Tactical variables
        self.current_cover = None
        self.flank_position = None
        self.ambush_position = None
        
    def calculate_genetic_fitness(self):
        fitness = (self.performance['kills'] * 2 + 
                  self.performance['assists'] * 1 +
                  self.performance['survival_time'] * 0.1 +
                  self.performance['communication_count'] * 0.5 -
                  self.performance['damage_taken'] * 0.5)
        return max(0, fitness)
    
    def mutate(self):
        self.genetic_traits = {k: max(0.1, min(0.95, v + random.uniform(-0.2, 0.2))) 
                              for k, v in self.genetic_traits.items()}
        self.neural_net.mutate()
    
    def crossover(self, other):
        child = AdvancedTacticalAI()
        
        # Crossover genetic traits
        for trait in self.genetic_traits:
            if random.random() > 0.5:
                child.genetic_traits[trait] = self.genetic_traits[trait]
            else:
                child.genetic_traits[trait] = other.genetic_traits[trait]
        
        # Crossover neural network
        child.neural_net = self.neural_net.crossover(other.neural_net)
        
        return child
    
    def communicate(self, entity, allies, game_map):
        current_time = time.time()
        if current_time - self.last_communication < 3.0 / self.genetic_traits['communication_frequency']:
            return
        
        communication_made = False
        
        for ally in allies:
            if ally == entity or not hasattr(ally, 'ai'):
                continue
            
            distance = math.sqrt((entity.x - ally.x)**2 + (entity.y - ally.y)**2)
            if distance <= self.communication_range and self.has_line_of_sight(entity.x, entity.y, ally.x, ally.y, game_map):
                
                # Share player position information
                if self.memory.last_known_positions:
                    last_pos = self.memory.last_known_positions[-1]
                    ally.ai.memory.add_memory(last_pos, 0.8, "radio_communication")
                
                # Share threat information
                for threat_pos, threat_info in self.known_threats.items():
                    ally.ai.known_threats[threat_pos] = threat_info
                
                # Coordinate squad actions
                if self.squad_role == AISquadRole.LEADER and len(self.squad_members) > 0:
                    self.coordinate_squad_attack(entity, allies)
                
                communication_made = True
        
        if communication_made:
            self.performance['communication_count'] += 1
            self.last_communication = current_time
    
    def has_line_of_sight(self, x1, y1, x2, y2, game_map, max_checks=8):
        dx = x2 - x1
        dy = y2 - y1
        distance = max(abs(dx), abs(dy))
        
        if distance == 0:
            return True
        
        for i in range(1, min(distance, max_checks) + 1):
            t = i / distance
            check_x = int(x1 + dx * t)
            check_y = int(y1 + dy * t)
            
            if (0 <= check_x < len(game_map[0]) and 0 <= check_y < len(game_map)):
                if game_map[check_y][check_x] in ['█', '▓', '#']:
                    return False
        
        return True
    
    def coordinate_squad_attack(self, entity, allies):
        if not self.memory.last_known_positions:
            return
        
        target_pos = self.memory.last_known_positions[-1]
        
        for member in self.squad_members:
            if hasattr(member, 'ai'):
                member.ai.assigned_target = target_pos
                
                # Assign flanking positions
                if member.ai.squad_role == AISquadRole.ASSAULT:
                    flank_x = target_pos[0] + random.randint(-3, 3)
                    flank_y = target_pos[1] + random.randint(-3, 3)
                    member.ai.flank_position = (flank_x, flank_y)
    
    def form_squad(self, entity, allies, max_size=4):
        if self.squad_role or len(self.squad_members) >= max_size:
            return
        
        nearby_allies = []
        for ally in allies:
            if (ally != entity and hasattr(ally, 'ai') and 
                ally.ai.squad_role is None and
                math.sqrt((entity.x - ally.x)**2 + (entity.y - ally.y)**2) < 10):
                nearby_allies.append(ally)
        
        if len(nearby_allies) >= 2:
            # This entity becomes squad leader
            self.squad_role = AISquadRole.LEADER
            
            # Select squad members
            self.squad_members = random.sample(nearby_allies, min(3, len(nearby_allies)))
            
            # Assign roles
            roles = [AISquadRole.ASSAULT, AISquadRole.SUPPORT, AISquadRole.MEDIC]
            for i, member in enumerate(self.squad_members):
                member.ai.squad_role = roles[i % len(roles)]
                member.ai.squad_members = [entity] + [m for m in self.squad_members if m != member]
    
    def analyze_tactical_situation(self, entity, player, allies, enemies, game_map, buildings):
        # Create input vector for neural network
        situation = np.zeros(24)
        
        # Health and status (0-3)
        situation[0] = entity.body.get_overall_health()
        situation[1] = entity.body.get_mobility()
        situation[2] = entity.body.get_combat_efficiency()
        situation[3] = entity.mental_state if hasattr(entity, 'mental_state') else 1.0
        
        # Ammo status (4)
        if entity.weapon:
            situation[4] = entity.weapon.ammunition.current_magazine / entity.weapon.ammunition.magazine_capacity
        else:
            situation[4] = 0
        
        # Player threat (5-7)
        if player:
            distance_to_player = math.sqrt((entity.x - player.x)**2 + (entity.y - player.y)**2)
            situation[5] = min(distance_to_player / 20.0, 1.0)
            situation[6] = 1.0 if player.body.get_overall_health() > 0.5 else 0.5
        else:
            situation[5] = 1.0
            situation[6] = 0
        
        # Squad status (7-9)
        situation[7] = len(self.squad_members) / 4.0
        situation[8] = 1.0 if self.squad_role == AISquadRole.LEADER else 0.0
        situation[9] = sum(1 for m in self.squad_members if m.body.get_overall_health() > 0.5) / max(1, len(self.squad_members))
        
        # Environmental factors (10-13)
        situation[10] = 1.0 if any(b.is_inside(entity.x, entity.y) for b in buildings) else 0.0
        situation[11] = self._calculate_cover_quality(entity.x, entity.y, game_map, buildings)
        situation[12] = len([e for e in enemies if math.sqrt((entity.x - e.x)**2 + (entity.y - e.y)**2) < 10]) / 5.0
        situation[13] = len(allies) / 8.0
        
        # Memory and prediction (14-17)
        if self.memory.last_known_positions:
            situation[14] = 1.0
            last_pos = self.memory.last_known_positions[-1]
            situation[15] = math.sqrt((entity.x - last_pos[0])**2 + (entity.y - last_pos[1])**2) / 30.0
        else:
            situation[14] = 0
            situation[15] = 1.0
        
        predicted_pos = self.memory.predict_player_position()
        if predicted_pos:
            situation[16] = 1.0
            situation[17] = math.sqrt((entity.x - predicted_pos[0])**2 + (entity.y - predicted_pos[1])**2) / 30.0
        else:
            situation[16] = 0
            situation[17] = 1.0
        
        # Genetic traits (18-23)
        situation[18] = self.genetic_traits['aggression']
        situation[19] = self.genetic_traits['caution']
        situation[20] = self.genetic_traits['teamwork']
        situation[21] = self.genetic_traits['accuracy']
        situation[22] = self.genetic_traits['mobility_preference']
        situation[23] = self.genetic_traits['communication_frequency']
        
        return situation
    
    def _calculate_cover_quality(self, x, y, game_map, buildings):
        cover_score = 0.0
        
        # Check building cover
        for building in buildings:
            if building.is_inside(x, y):
                cover_score += 0.7
                break
        
        # Check terrain cover
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            check_x, check_y = x + dx, y + dy
            if (0 <= check_x < len(game_map[0]) and 0 <= check_y < len(game_map)):
                if game_map[check_y][check_x] in ['█', '▓', '#']:
                    cover_score += 0.1
        
        return min(1.0, cover_score)
    
    def make_decision(self, entity, player, allies, enemies, game_map, buildings):
        # Update memory with player position
        if player and player.body.get_overall_health() > 0:
            self.memory.add_memory((player.x, player.y), 1.0, "visual")
            self.memory.player_pattern.append((player.x, player.y))
        
        # Communicate with allies
        self.communicate(entity, allies, game_map)
        
        # Form squad if needed
        if self.squad_role is None and random.random() < 0.1:
            self.form_squad(entity, allies)
        
        # Analyze situation
        situation = self.analyze_tactical_situation(entity, player, allies, enemies, game_map, buildings)
        
        # Get neural network prediction
        action_probs = self.neural_net.forward(situation)
        
        # Adjust probabilities based on genetic traits and current state
        if entity.body.get_overall_health() < 0.3:
            action_probs[2] *= 2.0  # Favor retreat
        if entity.weapon and entity.weapon.ammunition.current_magazine < 3:
            action_probs[4] *= 3.0  # Favor reload
        
        # Select action
        actions = ['attack', 'move_tactical', 'retreat', 'take_cover', 'reload', 'flank', 'support', 'ambush']
        chosen_action = random.choices(actions, weights=action_probs, k=1)[0]
        
        # Update state based on action
        if chosen_action in ['attack', 'flank', 'ambush']:
            self.state = AIState.COMBAT
        elif chosen_action == 'retreat':
            self.state = AIState.RETREAT
        elif chosen_action in ['move_tactical', 'take_cover']:
            self.state = AIState.PATROL
        elif chosen_action == 'support':
            self.state = AIState.SUPPORT
        
        return chosen_action
    
    def execute_action(self, entity, action, player, allies, enemies, game_map):
        if action == 'attack':
            return self._execute_attack(entity, player, game_map)
        elif action == 'move_tactical':
            return self._execute_move_tactical(entity, player, game_map)
        elif action == 'retreat':
            return self._execute_retreat(entity, player, game_map)
        elif action == 'take_cover':
            return self._execute_take_cover(entity, game_map, buildings)
        elif action == 'reload':
            return self._execute_reload(entity)
        elif action == 'flank':
            return self._execute_flank(entity, player, game_map)
        elif action == 'support':
            return self._execute_support(entity, allies, game_map)
        elif action == 'ambush':
            return self._execute_ambush(entity, player, game_map)
        
        return False
    
    def _execute_attack(self, entity, player, game_map):
        if not entity.weapon or not entity.weapon.can_fire() or not player:
            return False
        
        distance = math.sqrt((entity.x - player.x)**2 + (entity.y - player.y)**2)
        if distance <= entity.weapon.range:
            # Simple attack logic - in real implementation would use weapon system
            return True
        return False
    
    def _execute_move_tactical(self, entity, player, game_map):
        if not player:
            return self._execute_random_move(entity, game_map)
        
        # Move toward player or last known position
        target_x, target_y = player.x, player.y
        if self.memory.last_known_positions:
            target_x, target_y = self.memory.last_known_positions[-1]
        
        return self._move_toward(entity, target_x, target_y, game_map)
    
    def _execute_retreat(self, entity, player, game_map):
        if not player:
            return self._execute_random_move(entity, game_map)
        
        # Move away from player
        dx = entity.x - player.x
        dy = entity.y - player.y
        
        # Normalize and move
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx, dy = dx/distance, dy/distance
            new_x = int(entity.x + dx * 2)
            new_y = int(entity.y + dy * 2)
            
            if self._is_valid_position(new_x, new_y, game_map):
                entity.x, entity.y = new_x, new_y
                return True
        
        return self._execute_random_move(entity, game_map)
    
    def _execute_take_cover(self, entity, game_map, buildings):
        # Find nearest cover
        best_cover = None
        best_distance = float('inf')
        
        for building in buildings:
            dist = math.sqrt((entity.x - building.x)**2 + (entity.y - building.y)**2)
            if dist < best_distance:
                best_cover = building
                best_distance = dist
        
        if best_cover and best_distance < 15:
            return self._move_toward(entity, best_cover.x + best_cover.width//2, 
                                   best_cover.y + best_cover.height//2, game_map)
        
        return self._execute_random_move(entity, game_map)
    
    def _execute_reload(self, entity):
        if entity.weapon:
            return entity.weapon.reload()
        return False
    
    def _execute_flank(self, entity, player, game_map):
        if not player or not self.memory.last_known_positions:
            return self._execute_move_tactical(entity, player, game_map)
        
        # Calculate flanking position
        last_pos = self.memory.last_known_positions[-1]
        flank_x = last_pos[0] + random.randint(-5, 5)
        flank_y = last_pos[1] + random.randint(-5, 5)
        
        return self._move_toward(entity, flank_x, flank_y, game_map)
    
    def _execute_support(self, entity, allies, game_map):
        # Find nearest ally that needs help
        nearest_ally = None
        min_distance = float('inf')
        
        for ally in allies:
            if ally != entity and ally.body.get_overall_health() < 0.7:
                dist = math.sqrt((entity.x - ally.x)**2 + (entity.y - ally.y)**2)
                if dist < min_distance:
                    nearest_ally = ally
                    min_distance = dist
        
        if nearest_ally and min_distance < 15:
            return self._move_toward(entity, nearest_ally.x, nearest_ally.y, game_map)
        
        return self._execute_random_move(entity, game_map)
    
    def _execute_ambush(self, entity, player, game_map):
        if not self.memory.predict_player_position():
            return self._execute_move_tactical(entity, player, game_map)
        
        # Move to predicted player position
        pred_pos = self.memory.predict_player_position()
        return self._move_toward(entity, pred_pos[0], pred_pos[1], game_map)
    
    def _execute_random_move(self, entity, game_map):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_x, new_y = entity.x + dx, entity.y + dy
            if self._is_valid_position(new_x, new_y, game_map):
                entity.x, entity.y = new_x, new_y
                return True
        return False
    
    def _move_toward(self, entity, target_x, target_y, game_map):
        dx = target_x - entity.x
        dy = target_y - entity.y
        
        if dx == 0 and dy == 0:
            return False
        
        # Normalize direction
        distance = math.sqrt(dx*dx + dy*dy)
        dx, dy = dx/distance, dy/distance
        
        # Try to move in the general direction
        move_x = int(entity.x + (1 if dx > 0 else -1))
        move_y = int(entity.y + (1 if dy > 0 else -1))
        
        if self._is_valid_position(move_x, entity.y, game_map):
            entity.x = move_x
            return True
        elif self._is_valid_position(entity.x, move_y, game_map):
            entity.y = move_y
            return True
        
        return self._execute_random_move(entity, game_map)
    
    def _is_valid_position(self, x, y, game_map):
        if x < 0 or y < 0 or y >= len(game_map) or x >= len(game_map[0]):
            return False
        
        return game_map[y][x] not in ['█', '▓', '#', '~']

# =============================
# Evolution System
# =============================

class EvolutionSystem:
    def __init__(self):
        self.generation = 0
        self.genome_pool = []
        self.performance_history = []
        self.mutation_rate = 0.2
        self.crossover_rate = 0.6
    
    def evaluate_generation(self, enemies):
        successful_ais = []
        
        for enemy in enemies:
            if hasattr(enemy, 'ai'):
                fitness = enemy.ai.calculate_genetic_fitness()
                
                if fitness > 5:  # Threshold for successful AI
                    successful_ais.append((enemy.ai, fitness))
                
                # Track performance
                self.performance_history.append({
                    'genome_id': enemy.ai.genome_id,
                    'fitness': fitness,
                    'generation': self.generation,
                    'traits': enemy.ai.genetic_traits.copy()
                })
        
        # Keep top performers
        successful_ais.sort(key=lambda x: x[1], reverse=True)
        top_performers = [ai for ai, fitness in successful_ais[:10]]
        
        self.genome_pool.extend(top_performers)
        
        # Keep only the best 20 genomes
        if len(self.genome_pool) > 20:
            self.genome_pool.sort(key=lambda ai: ai.calculate_genetic_fitness(), reverse=True)
            self.genome_pool = self.genome_pool[:20]
    
    def create_next_generation(self, count):
        self.generation += 1
        
        new_ais = []
        
        while len(new_ais) < count:
            if len(self.genome_pool) >= 2 and random.random() < self.crossover_rate:
                # Crossover
                parent1, parent2 = random.sample(self.genome_pool, 2)
                child = parent1.crossover(parent2)
                
                if random.random() < self.mutation_rate:
                    child.mutate()
                
                new_ais.append(child)
            elif self.genome_pool:
                # Mutation of existing
                parent = random.choice(self.genome_pool)
                child = AdvancedTacticalAI()
                child.genetic_traits = parent.genetic_traits.copy()
                child.neural_net = parent.neural_net
                child.mutate()
                new_ais.append(child)
            else:
                # New random AI
                new_ais.append(AdvancedTacticalAI())
        
        return new_ais
    
    def get_generation_stats(self):
        if not self.performance_history:
            return {}
        
        recent_perf = [p for p in self.performance_history if p['generation'] >= self.generation - 1]
        if not recent_perf:
            return {}
        
        avg_fitness = sum(p['fitness'] for p in recent_perf) / len(recent_perf)
        max_fitness = max(p['fitness'] for p in recent_perf)
        
        return {
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'pool_size': len(self.genome_pool)
        }

# =============================
# Enhanced Body System
# =============================

class BodyPart:
    def __init__(self, name, max_health, armor_slot, critical_multiplier, functionality_impact):
        self.name = name
        self.health = max_health
        self.max_health = max_health
        self.armor_slot = armor_slot
        self.critical_multiplier = critical_multiplier
        self.functionality_impact = functionality_impact
        self.wounded = False
        self.fracture = False
        self.bleeding = False
        self.pain_level = 0.0
        
    def take_damage(self, damage, damage_type='ballistic'):
        original_health = self.health
        self.health = max(0, self.health - damage)
        
        self.pain_level = min(1.0, self.pain_level + (damage / self.max_health) * 2)
        
        if self.health < self.max_health * 0.7:
            self.wounded = True
        
        if damage > 15 and random.random() < 0.3:
            self.fracture = True
        
        if damage_type in ['ballistic', 'sharp'] and damage > 10 and random.random() < 0.4:
            self.bleeding = True
        
        return original_health - self.health
    
    def heal(self, amount):
        old_health = self.health
        self.health = min(self.max_health, self.health + amount)
        
        self.pain_level = max(0, self.pain_level - (amount / self.max_health))
        
        if self.health > self.max_health * 0.8:
            self.wounded = False
        
        if self.health > self.max_health * 0.9 and random.random() < 0.5:
            self.fracture = False
        
        return self.health - old_health
    
    def stop_bleeding(self):
        self.bleeding = False
    
    def reduce_pain(self, amount):
        self.pain_level = max(0, self.pain_level - amount)
    
    def get_functionality(self):
        functionality = self.health / self.max_health
        
        if self.fracture:
            functionality *= 0.3
        if self.wounded:
            functionality *= 0.7
        if self.bleeding:
            functionality *= 0.8
        if self.pain_level > 0.5:
            functionality *= (1 - self.pain_level * 0.5)
            
        return max(0.1, functionality)
    
    def get_status(self):
        status = f"{self.health}/{self.max_health}"
        if self.bleeding:
            status += " 💧"
        if self.fracture:
            status += " 🦴"
        elif self.wounded:
            status += " 🩹"
        if self.pain_level > 0.7:
            status += " 🔥"
        elif self.pain_level > 0.3:
            status += " 💢"
        return status

class Body:
    def __init__(self):
        self.parts = {
            'head': BodyPart("Голова", 40, 'helmet', 2.5, 0.8),
            'torso': BodyPart("Торс", 100, 'vest', 1.5, 0.6),
            'left_arm': BodyPart("Левая рука", 60, None, 1.0, 0.4),
            'right_arm': BodyPart("Правая рука", 60, None, 1.0, 0.4),
            'left_leg': BodyPart("Левая нога", 70, None, 1.0, 0.5),
            'right_leg': BodyPart("Правая нога", 70, None, 1.0, 0.5)
        }
        
        self.armor = {
            'helmet': None,
            'vest': None
        }
    
    def take_damage(self, damage, body_part_name, damage_type='ballistic'):
        part = self.parts[body_part_name]
        
        armor = self.armor.get(part.armor_slot) if part.armor_slot else None
        if armor:
            damage = armor.absorb_damage(damage, damage_type)
        
        actual_damage = part.take_damage(damage, damage_type)
        
        critical_damage = 0
        if random.random() < 0.1:
            critical_damage = actual_damage * (part.critical_multiplier - 1)
            part.take_damage(critical_damage, damage_type)
        
        return actual_damage + critical_damage
    
    def get_random_body_part(self, hit_distribution=None):
        if not hit_distribution:
            hit_distribution = {
                'head': 0.1,
                'torso': 0.5,
                'left_arm': 0.08,
                'right_arm': 0.08,
                'left_leg': 0.12,
                'right_leg': 0.12
            }
        
        parts = list(hit_distribution.keys())
        probabilities = list(hit_distribution.values())
        return random.choices(parts, weights=probabilities, k=1)[0]
    
    def get_overall_health(self):
        total_health = sum(part.health for part in self.parts.values())
        total_max_health = sum(part.max_health for part in self.parts.values())
        return total_health / total_max_health
    
    def get_bleeding_count(self):
        return sum(1 for part in self.parts.values() if part.bleeding)
    
    def get_fracture_count(self):
        return sum(1 for part in self.parts.values() if part.fracture)
    
    def get_pain_level(self):
        return max(part.pain_level for part in self.parts.values())
    
    def get_mobility(self):
        leg_functionality = (self.parts['left_leg'].get_functionality() + 
                           self.parts['right_leg'].get_functionality()) / 2
        torso_functionality = self.parts['torso'].get_functionality()
        return min(leg_functionality, torso_functionality)
    
    def get_combat_efficiency(self):
        arm_functionality = (self.parts['left_arm'].get_functionality() + 
                           self.parts['right_arm'].get_functionality()) / 2
        head_functionality = self.parts['head'].get_functionality()
        return min(arm_functionality, head_functionality)

# =============================
# Enhanced Soldier with Advanced AI
# =============================

class EnhancedSoldier:
    def __init__(self, x, y, name, rank, unit, hp=100, stamina=100):
        self.x = x
        self.y = y
        self.name = name
        self.rank = rank
        self.unit = unit
        
        self.body = Body()
        self.stamina = stamina
        self.max_stamina = stamina
        self.mental_state = 1.0
        
        self.weapon = None
        self.ai = AdvancedTacticalAI()
        
        self.inventory = AdvancedInventory(30)
        self.medical_supplies = 0
        
        self.skills = {
            'marksmanship': random.uniform(0.5, 1.0),
            'tactics': random.uniform(0.5, 1.0),
            'stealth': random.uniform(0.5, 1.0),
            'endurance': random.uniform(0.5, 1.0),
            'medicine': random.uniform(0.3, 0.8)
        }
        
        self.char = 'S'
        self.color = Colors.BLUE
        self.is_medic = False
        
        self.damage_taken = 0
        self.damage_dealt = 0
        self.enemies_killed = 0
        self.spawn_time = time.time()
        
    def get_effective_health(self):
        base_health = self.body.get_overall_health()
        mobility = self.body.get_mobility()
        efficiency = self.body.get_combat_efficiency()
        return base_health * mobility * efficiency
    
    def take_damage(self, damage, body_part_name=None, damage_type='ballistic'):
        if not body_part_name:
            body_part_name = self.body.get_random_body_part()
        
        actual_damage = self.body.take_damage(damage, body_part_name, damage_type)
        self.damage_taken += actual_damage
        
        if actual_damage > 15:
            self.mental_state -= 0.1
        
        if self.body.get_overall_health() <= 0:
            return True
        
        return False
    
    def update(self, delta_time):
        self.stamina = min(self.max_stamina, self.stamina + 10 * delta_time)
        
        bleeding_count = self.body.get_bleeding_count()
        if bleeding_count > 0:
            bleed_damage = bleeding_count * 2 * delta_time
            self.body.parts['torso'].take_damage(bleed_damage)
        
        self.mental_state = min(1.0, self.mental_state + 0.1 * delta_time)
        
        if bleeding_count > 0 and random.random() < 0.05:
            self.auto_treat_bleeding()
    
    def auto_treat_bleeding(self):
        if self.body.get_bleeding_count() > 0 and self.medical_supplies > 0:
            for item in self.inventory.items:
                if hasattr(item, 'item_type') and item.item_type == 'bandage' and item.can_use():
                    for part_name, part in self.body.parts.items():
                        if part.bleeding:
                            success, result = self.heal(item, part_name)
                            if success:
                                return True
        return False

class AdvancedMedicEnemy(EnhancedSoldier):
    def __init__(self, x, y, name, rank, unit):
        super().__init__(x, y, name, rank, unit)
        self.char = 'M'
        self.color = Colors.PURPLE
        self.is_medic = True
        self.healing_range = 6
        
        self.skills['medicine'] = random.uniform(0.8, 1.0)
        self._add_medical_supplies()
        self.ai.squad_role = AISquadRole.MEDIC
    
    def _add_medical_supplies(self):
        supplies_count = random.randint(4, 8)
        for _ in range(supplies_count):
            item_type = random.choice(['bandage', 'medkit', 'painkiller', 'splint'])
            # In full implementation, would use medical generator
            self.medical_supplies += 1
    
    def medic_ai(self, allies, game_map):
        if self.medical_supplies <= 0:
            return False
            
        best_target = None
        best_priority = -1
        
        for ally in allies:
            if ally == self or ally.body.get_overall_health() <= 0:
                continue
                
            distance = math.sqrt((self.x - ally.x)**2 + (self.y - ally.y)**2)
            if distance <= self.healing_range:
                has_los = self.has_line_of_sight(self.x, self.y, ally.x, ally.y, game_map)
                priority = self._calculate_healing_priority(ally, distance, has_los)
                
                if priority > best_priority:
                    best_priority = priority
                    best_target = ally
        
        if best_target and best_priority > 0.3:
            return self._perform_healing(best_target)
            
        return False
    
    def _calculate_healing_priority(self, ally, distance, has_los):
        priority = 0.0
        
        health_ratio = ally.body.get_overall_health()
        bleeding_count = ally.body.get_bleeding_count()
        fracture_count = ally.body.get_fracture_count()
        
        priority += bleeding_count * 0.4
        priority += (1.0 - health_ratio) * 0.3
        priority += fracture_count * 0.2
        
        if not has_los:
            priority *= 0.7
            
        distance_penalty = distance / self.healing_range
        priority *= (1.0 - distance_penalty * 0.3)
        
        if hasattr(ally, 'ai') and ally.ai.squad_role == AISquadRole.LEADER:
            priority *= 1.3
            
        return priority
    
    def _perform_healing(self, target):
        # Simplified healing logic
        if target.body.get_bleeding_count() > 0:
            for part_name, part in target.body.parts.items():
                if part.bleeding:
                    part.stop_bleeding()
                    self.medical_supplies -= 1
                    return True
        
        elif target.body.get_overall_health() < 0.8:
            healing_amount = 20
            for part in target.body.parts.values():
                part.heal(healing_amount * 0.3)
            self.medical_supplies -= 1
            return True
            
        return False
    
    def has_line_of_sight(self, x1, y1, x2, y2, game_map):
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        return distance <= self.healing_range

# =============================
# Supporting Classes
# =============================

class AdvancedInventory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []
        
    def add_item(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
            return True
        return False

class Armor:
    def __init__(self, name, armor_type, protection_level, durability, weight, coverage):
        self.name = name
        self.armor_type = armor_type
        self.protection_level = protection_level
        self.durability = durability
        self.max_durability = durability
        self.weight = weight
        self.coverage = coverage
        
        self.damage_resistance = {
            'ballistic': protection_level * 0.8,
            'explosive': protection_level * 0.5,
            'sharp': protection_level * 0.6,
            'blunt': protection_level * 0.9
        }
    
    def absorb_damage(self, damage, damage_type='ballistic'):
        if self.durability <= 0:
            return damage
        
        resistance = self.damage_resistance.get(damage_type, self.protection_level * 0.7)
        absorbed = min(damage * (resistance / 10.0), damage * 0.8)
        
        if random.random() > self.coverage:
            absorbed *= 0.5
        
        damage_to_armor = min(absorbed * 0.1, self.durability)
        self.durability -= damage_to_armor
        
        return damage - absorbed

class MedicalItem:
    def __init__(self, name, item_type, healing_power, uses, weight, special_effects=None):
        self.name = name
        self.item_type = item_type
        self.healing_power = healing_power
        self.max_uses = uses
        self.remaining_uses = uses
        self.weight = weight
        self.special_effects = special_effects or []
    
    def use(self, target_body, body_part_name=None):
        if self.remaining_uses <= 0:
            return False, "Предмет израсходован"
        
        success = True
        result = ""
        
        if self.item_type == 'bandage':
            if body_part_name:
                part = target_body.parts[body_part_name]
                if part.bleeding:
                    part.stop_bleeding()
                    result = f"Кровотечение на {part.name} остановлено"
                else:
                    success = False
                    result = f"На {part.name} нет кровотечения"
            else:
                success = False
                result = "Для бинта нужно указать часть тела"
        
        elif self.item_type == 'medkit':
            total_healed = 0
            for part_name, part in target_body.parts.items():
                healed = part.heal(self.healing_power)
                total_healed += healed
            result = f"Восстановлено {total_healed} HP"
        
        if success:
            self.remaining_uses -= 1
        
        return success, result
    
    def can_use(self):
        return self.remaining_uses > 0

class EnhancedModernWeapon:
    def __init__(self, name, weapon_type, damage, fire_rate, accuracy, range, ammo_type='ball'):
        self.name = name
        self.weapon_type = weapon_type
        self.damage = damage
        self.fire_rate = fire_rate
        self.accuracy = accuracy
        self.range = range
        self.spread = 0.1
        
        self.current_ammo_type = ammo_type
        self.ammunition = Ammunition(ammo_type, 1.0, 0.5, 1.0, 1.0)
        
        self.last_shot_time = 0
        self.shots_fired = 0
        self.shots_hit = 0
    
    def can_fire(self):
        current_time = time.time()
        time_since_last_shot = current_time - self.last_shot_time
        return (self.ammunition.current_magazine > 0 and 
                time_since_last_shot >= (1.0 / self.fire_rate))
    
    def fire(self, shooter, target_pos, distance, cover_bonus=0):
        if not self.can_fire():
            return None
        
        hit_chance = max(0.1, self.accuracy * (1.0 - self.spread) * (1.0 - cover_bonus))
        hit_roll = random.random()
        hit_success = hit_roll <= hit_chance
        
        if not hit_success:
            return {'hit': False, 'damage': 0, 'critical': False}
        
        base_damage = self.damage * self.ammunition.damage_modifier
        critical = random.random() < 0.05
        
        if critical:
            base_damage *= 1.5
        
        damage = int(base_damage * random.uniform(0.8, 1.2))
        
        self.shots_fired += 1
        if hit_success:
            self.shots_hit += 1
        
        self.ammunition.use_ammo()
        self.last_shot_time = time.time()
        
        return {
            'hit': True,
            'damage': damage,
            'critical': critical,
            'ammo_type': self.current_ammo_type
        }
    
    def reload(self):
        return self.ammunition.reload()
    
    def get_accuracy(self):
        if self.shots_fired == 0:
            return 0.0
        return self.shots_hit / self.shots_fired

class Ammunition:
    def __init__(self, ammo_type, damage_modifier, penetration, spread, range_modifier):
        self.ammo_type = ammo_type
        self.damage_modifier = damage_modifier
        self.penetration = penetration
        self.spread = spread
        self.range_modifier = range_modifier
        
        self.magazines = 7
        self.magazine_capacity = 30
        self.current_magazine = 30
        self.total_ammo = 210
    
    def use_ammo(self, count=1):
        if self.current_magazine >= count:
            self.current_magazine -= count
            self.total_ammo -= count
            return True
        return False
    
    def reload(self):
        if self.magazines > 0 and self.current_magazine < self.magazine_capacity:
            self.magazines -= 1
            self.current_magazine = self.magazine_capacity
            return True
        return False

class Building:
    def __init__(self, x, y, width, height, building_type, name):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.building_type = building_type
        self.name = name
        self.cover_bonus = 0.7 if building_type in ['bunker', 'fortification'] else 0.4
        self.char = '█' if building_type in ['barracks', 'bunker'] else '▄'
        self.color = Colors.ORANGE if building_type == 'barracks' else Colors.GRAY
    
    def is_inside(self, x, y):
        return (self.x <= x < self.x + self.width and 
                self.y <= y < self.y + self.height)

# =============================
# Complete Modern Warfare Game
# =============================

class CompleteModernWarfareGame:
    def __init__(self, screen_width=80, screen_height=25):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.map_width = 200
        self.map_height = 200
        self.camera_x = 0
        self.camera_y = 0
        
        self.player = None
        self.game_map = None
        self.entities = []
        self.buildings = []
        self.medical_items_on_ground = []
        self.weapon_items_on_ground = []
        self.armor_items_on_ground = []
        
        self.ui_state = 'game'
        self.selected_body_part = 'torso'
        self.selected_medical_item = None
        self.aiming_direction = None
        
        self.running = True
        self.last_update_time = time.time()
        self.game_time = 0
        self.messages = deque(maxlen=5)
        self.player_kills = 0
        
        self.evolution_system = EvolutionSystem()
        self.wave_number = 1
        self.enemies_per_wave = 25
        
        self.mission_objectives = {
            'eliminate_enemies': 25,
            'survive_time': 600
        }
        
        self._setup_terminal()
        self.generate_modern_battlefield()
    
    def _setup_terminal(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    
    def _restore_terminal(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def _get_input(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
    
    def add_message(self, message):
        self.messages.append(message)
    
    def generate_modern_battlefield(self):
        self.game_map = [['.' for _ in range(self.map_width)] for _ in range(self.map_height)]
        
        # Add some basic terrain
        for i in range(20):
            x, y = random.randint(0, self.map_width-1), random.randint(0, self.map_height-1)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                        if random.random() < 0.7:
                            self.game_map[ny][nx] = 'T'
        
        # Add buildings
        for i in range(8):
            x, y = random.randint(10, self.map_width-20), random.randint(10, self.map_height-20)
            width, height = random.randint(5, 12), random.randint(4, 8)
            building_type = random.choice(['barracks', 'bunker'])
            building = Building(x, y, width, height, building_type, f"Building {i+1}")
            self.buildings.append(building)
            
            for by in range(height):
                for bx in range(width):
                    if 0 <= x+bx < self.map_width and 0 <= y+by < self.map_height:
                        self.game_map[y+by][x+bx] = building.char
        
        # Place player
        start_x, start_y = self._find_safe_start_position()
        self.player = EnhancedSoldier(start_x, start_y, "Командир", "Капитан", "Альфа")
        
        # Give starting equipment
        self._give_starting_equipment()
        
        # Spawn first wave
        self.spawn_new_wave()
        
        self._update_camera()
        self.add_message("Миссия начата! Уничтожьте врагов и выживите.")
    
    def _give_starting_equipment(self):
        starter_weapon = EnhancedModernWeapon("M4A1", "assault_rifle", 30, 0.9, 0.7, 10)
        self.player.equip_weapon(starter_weapon)
    
    def spawn_new_wave(self):
        # Remove dead enemies
        self.entities = [e for e in self.entities if e == self.player or (hasattr(e, 'is_medic') and e.is_medic)]
        
        # Evaluate previous generation and create new one
        if self.wave_number > 1:
            self.evolution_system.evaluate_generation(self.entities)
        
        # Create new enemies with evolved AI
        new_enemies = self._generate_wave_enemies(self.enemies_per_wave)
        self.entities.extend(new_enemies)
        
        self.wave_number += 1
        self.enemies_per_wave = int(25 * (1 + self.wave_number * 0.15))
        self.add_message(f"Волна {self.wave_number}! Враги адаптировались...")
    
    def _generate_wave_enemies(self, count):
        enemies = []
        new_ais = self.evolution_system.create_next_generation(count)
        
        medics_count = max(2, count // 8)
        
        for i in range(count - medics_count):
            x, y = self._find_enemy_position()
            enemy = EnhancedSoldier(x, y, f"Враг W{self.wave_number}-{i+1}", "Рядовой", f"Волна {self.wave_number}")
            enemy.char = 'E'
            enemy.color = Colors.RED
            
            # Assign evolved AI
            if i < len(new_ais):
                enemy.ai = new_ais[i]
            
            enemy.weapon = EnhancedModernWeapon("AK-47", "assault_rifle", 28, 0.8, 0.65, 9)
            enemies.append(enemy)
        
        # Add medics
        for i in range(medics_count):
            x, y = self._find_enemy_position()
            medic = AdvancedMedicEnemy(x, y, f"Медик W{self.wave_number}-{i+1}", "Сержант", "Медслужба")
            if len(new_ais) > count - medics_count + i:
                medic.ai = new_ais[count - medics_count + i]
            enemies.append(medic)
            
        return enemies
    
    def _find_enemy_position(self):
        for _ in range(100):
            x = random.randint(0, self.map_width - 1)
            y = random.randint(0, self.map_height - 1)
            
            if (self.game_map[y][x] == '.' and 
                not any(e.x == x and e.y == y for e in self.entities) and
                (x != self.player.x or y != self.player.y)):
                return x, y
        
        return random.randint(0, self.map_width - 1), random.randint(0, self.map_height - 1)
    
    def _find_safe_start_position(self):
        for _ in range(100):
            x = random.randint(20, self.map_width - 20)
            y = random.randint(20, self.map_height - 20)
            
            if self.game_map[y][x] == '.':
                return x, y
        
        return self.map_width // 2, self.map_height // 2
    
    def _update_camera(self):
        self.camera_x = max(0, min(self.player.x - self.screen_width // 2, 
                                 self.map_width - self.screen_width))
        self.camera_y = max(0, min(self.player.y - self.screen_height // 2, 
                                 self.map_height - self.screen_height))
    
    def update_game(self, delta_time):
        self.game_time += delta_time
        
        # Check mission complete
        enemies_remaining = len([e for e in self.entities if e != self.player and e.body.get_overall_health() > 0])
        if enemies_remaining == 0:
            self.evolution_system.evaluate_generation(self.entities)
            self.spawn_new_wave()
        
        # Check player death
        if self.player.body.get_overall_health() <= 0:
            self.add_message("Вы погибли! Миссия провалена.")
            self.running = False
            return
        
        # Update player
        self.player.update(delta_time)
        
        # Update enemies
        for entity in self.entities[:]:
            if entity == self.player:
                continue
                
            if entity.body.get_overall_health() <= 0:
                if entity in self.entities:
                    self.entities.remove(entity)
                continue
            
            entity.update(delta_time)
            
            # Medic AI
            if entity.is_medic:
                entity.medic_ai(self.entities, self.game_map)
            
            # Tactical AI decision making
            if hasattr(entity, 'ai'):
                action = entity.ai.make_decision(entity, self.player, self.entities, 
                                               [e for e in self.entities if e != entity and e != self.player],
                                               self.game_map, self.buildings)
                
                entity.ai.execute_action(entity, action, self.player, self.entities, 
                                       [e for e in self.entities if e != entity and e != self.player], 
                                       self.game_map)
                
                # Enemy attacking logic
                if action == 'attack' and entity.weapon and entity.weapon.can_fire():
                    distance = math.sqrt((entity.x - self.player.x)**2 + (entity.y - self.player.y)**2)
                    if distance <= entity.weapon.range:
                        shot_result = entity.weapon.fire(entity, (self.player.x, self.player.y), distance, 0)
                        if shot_result and shot_result['hit']:
                            body_part = self.player.body.get_random_body_part()
                            self.player.take_damage(shot_result['damage'], body_part, 'ballistic')
                            self.add_message(f"{entity.name} попал вам в {body_part}! {shot_result['damage']} урона")
                            
                            # Update AI performance
                            entity.ai.performance['damage_dealt'] += shot_result['damage']
    
    def render_game_screen(self):
        os.system('clear')
        
        # Top panel
        health_percent = self.player.body.get_overall_health()
        health_bar = self._create_health_bar(health_percent)
        
        print(f"{Colors.BOLD}{Colors.CYAN}╔{'═' * (self.screen_width - 2)}╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║ {Colors.WHITE}{self.player.name} {Colors.YELLOW}[{self.player.rank}] {Colors.GREEN}{self.player.unit}{' ' * (self.screen_width - 40)}║{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║ {Colors.RED}Здоровье: {health_bar}{' ' * 30}║{Colors.RESET}")
        
        # Wave and evolution info
        wave_info = f"Волна: {self.wave_number} Врагов: {len([e for e in self.entities if e != self.player and e.body.get_overall_health() > 0])}"
        print(f"{Colors.BOLD}{Colors.CYAN}║ {Colors.WHITE}{wave_info}{' ' * (self.screen_width - len(wave_info) - 4)}║{Colors.RESET}")
        
        print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * (self.screen_width - 2)}╝{Colors.RESET}")
        
        # Game map
        map_display_height = self.screen_height - 10
        for display_y in range(map_display_height):
            actual_y = self.camera_y + display_y
            if actual_y >= self.map_height:
                break
                
            line = ""
            for display_x in range(self.screen_width):
                actual_x = self.camera_x + display_x
                if actual_x >= self.map_width:
                    break
                
                # Check for entities
                entity_char = None
                entity_color = Colors.WHITE
                
                if self.player.x == actual_x and self.player.y == actual_y:
                    entity_char = '@'
                    entity_color = Colors.BOLD + Colors.CYAN
                else:
                    for entity in self.entities:
                        if entity.x == actual_x and entity.y == actual_y and entity.body.get_overall_health() > 0:
                            entity_char = entity.char
                            entity_color = entity.color
                            break
                
                if entity_char:
                    line += f"{entity_color}{entity_char}{Colors.RESET}"
                else:
                    # Check for buildings
                    in_building = False
                    for building in self.buildings:
                        if building.is_inside(actual_x, actual_y):
                            line += f"{building.color}{building.char}{Colors.RESET}"
                            in_building = True
                            break
                    
                    if not in_building:
                        terrain_char = self.game_map[actual_y][actual_x]
                        terrain_color = self._get_terrain_color(terrain_char)
                        line += f"{terrain_color}{terrain_char}{Colors.RESET}"
            
            print(line)
        
        # Bottom panel
        print(f"{Colors.BOLD}{Colors.CYAN}╔{'═' * (self.screen_width - 2)}╗{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}║ {Colors.WHITE}WASD{Colors.RESET}-движение {Colors.WHITE}Пробел{Colors.RESET}-стрельба {Colors.WHITE}R{Colors.RESET}-перезарядка {Colors.WHITE}Q{Colors.RESET}-выход ║")
        print(f"{Colors.BOLD}{Colors.CYAN}╚{'═' * (self.screen_width - 2)}╝{Colors.RESET}")
        
        # Messages
        for msg in list(self.messages)[-3:]:
            print(f"{Colors.WHITE}{msg}{Colors.RESET}")
    
    def _create_health_bar(self, health_percent, length=20):
        filled = int(length * health_percent)
        bar = f"{Colors.RED}{'█' * filled}{Colors.GRAY}{'░' * (length - filled)}{Colors.RESET}"
        return f"{bar} {health_percent:.1%}"
    
    def _get_terrain_color(self, terrain_char):
        colors = {
            '.': Colors.GREEN,
            'T': Colors.DARK_GREEN,
            '█': Colors.ORANGE,
            '▄': Colors.GRAY
        }
        return colors.get(terrain_char, Colors.WHITE)
    
    def handle_game_input(self, key):
        if key in ['w', 'a', 's', 'd']:
            self._handle_movement(key)
        elif key == ' ':
            self._handle_shooting()
        elif key == 'r':
            self._handle_reload()
        elif key == 'q':
            self.running = False
    
    def _handle_movement(self, key):
        dx, dy = 0, 0
        
        if key == 'w':
            dy = -1
        elif key == 's':
            dy = 1
        elif key == 'a':
            dx = -1
        elif key == 'd':
            dx = 1
        
        mobility = self.player.body.get_mobility()
        if mobility < 0.5 and random.random() < 0.3:
            self.add_message("Ранение замедляет вас!")
            return
        
        new_x, new_y = self.player.x + dx, self.player.y + dy
        
        if (0 <= new_x < self.map_width and 0 <= new_y < self.map_height and
            self.game_map[new_y][new_x] not in ['█', '▄', 'T']):
            
            self.player.x, self.player.y = new_x, new_y
            self.player.stamina = max(0, self.player.stamina - 2)
            self._update_camera()
    
    def _handle_shooting(self):
        if not self.player.weapon or not self.player.weapon.can_fire():
            self.add_message("Нельзя стрелять!")
            return
        
        # Find nearest enemy in front of player
        nearest_enemy = None
        min_distance = float('inf')
        
        for entity in self.entities:
            if entity == self.player or entity.body.get_overall_health() <= 0:
                continue
            
            distance = math.sqrt((self.player.x - entity.x)**2 + (self.player.y - entity.y)**2)
            if distance < min_distance and distance <= self.player.weapon.range:
                nearest_enemy = entity
                min_distance = distance
        
        if nearest_enemy:
            shot_result = self.player.weapon.fire(self.player, (nearest_enemy.x, nearest_enemy.y), min_distance, 0)
            if shot_result and shot_result['hit']:
                body_part = nearest_enemy.body.get_random_body_part()
                if nearest_enemy.take_damage(shot_result['damage'], body_part, 'ballistic'):
                    self.add_message(f"{nearest_enemy.name} уничтожен!")
                    self.player_kills += 1
                    self.player.enemies_killed += 1
                else:
                    self.add_message(f"Попадание в {nearest_enemy.name}! {shot_result['damage']} урона")
            else:
                self.add_message("Промах!")
        else:
            self.add_message("Нет целей в радиусе поражения!")
    
    def _handle_reload(self):
        if self.player.weapon:
            if self.player.weapon.reload():
                self.add_message("Перезарядка!")
            else:
                self.add_message("Нет магазинов для перезарядки!")
        else:
            self.add_message("Нет оружия!")
    
    def run(self):
        try:
            while self.running:
                current_time = time.time()
                delta_time = current_time - self.last_update_time
                
                if delta_time < 0.05:
                    time.sleep(0.05 - delta_time)
                    continue
                
                self.last_update_time = current_time
                
                if self.ui_state == 'game':
                    self.render_game_screen()
                    self.update_game(delta_time)
                
                key = self._get_input()
                if key:
                    if self.ui_state == 'game':
                        self.handle_game_input(key)
                
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._restore_terminal()

# =============================
# Main Execution
# =============================

if __name__ == '__main__':
    try:
        columns, rows = os.get_terminal_size()
        if columns < 80 or rows < 25:
            print("Рекомендуемый размер терминала: 80x25")
            print(f"Текущий размер: {columns}x{rows}")
    except:
        pass
    
    game = CompleteModernWarfareGame()
    game.run()
