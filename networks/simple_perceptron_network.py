from .base_network import Base_Network
from ..neurons.signal_neuron import Signal_Neuron
import numpy as np

class Simple_Perceptron_Network(Base_Network):
  def __init__(self, input_layer_size, output_layer_size=1, max_generations=2000):
    self.input_layer_size = input_layer_size
    self.output_layer_size = output_layer_size
    self.input_layer = None 
    self.output = None
    self.desired_output = None
    self.output_neuron = np.array([Signal_Neuron(input_layer_size) for i in range(output_layer_size)])
    self.training_coefficient = 0.01
    self.max_generations = max_generations
    self.actual_generation = 1

  def add_output_neuron(self, number=1):
    for i in range(number):
      self.output_neuron = np.append(self.output_neuron, Signal_Neuron(self.input_layer_size))
    self.output_layer_size += number
