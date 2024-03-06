from .neuron import Neuron

class Signal_Neuron(Neuron):
  def _act_function(self, u):
    if u >= 0: return 1
    else: return -1
