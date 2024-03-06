from .neuron import Neuron

class Binary_Neuron(Neuron):
  def _act_function(self, u):
    if u >= 0: return 1
    else: return 0
