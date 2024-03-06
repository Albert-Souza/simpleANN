import numpy as np

class Neuron:
  def __init__(self, size):
    self.input = np.array([-1])
    self.input = np.append(self.input, np.empty(size))
    self.weight = np.random.uniform(size=size+1)
    self.u = None
    self.output = None

  def add_inputs(self, input):
    if len(list) != (len(self.weight)):
      print('Erro: quantidade de inputs e pesos são diferentes!')
    else:
      self.input = np.append(self.input, input)

  def set_weight(self, weight):
    self.weight = weight

  def _act_function(self, u):
    return u

  def __calc__(self):
    self.u = np.dot(self.input, self.weight)
    self.output = self._act_function(self.u)