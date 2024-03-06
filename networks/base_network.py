from ..neurons.neuron import Neuron
import numpy as np

class Base_Network:
  def __init__(self, input_layer_size, output_layer_size=0, max_generations=2000):
    self.input_layer_size = input_layer_size
    self.output_layer_size = output_layer_size
    self.input_layer = None 
    self.output = None
    self.desired_output = None
    self.output_neuron = np.array([Neuron(input_layer_size) for _ in range(output_layer_size)])
    self.training_coefficient = 0.01
    self.max_generations = max_generations
    self.actual_generation = 1

  def set_inputs(self, input):
    if len(input[0]) != self.input_layer_size:
      print('Erro: quantidade de inputs e tamanho da input layer são diferentes!')
    else:
      self.input_layer = np.array(input)
  
  def set_desired_output(self, desired_output):
    if len(self.input_layer) != len(desired_output):
      print('Erro: quantidade de amostras são discrepantes com o input!')
    else: 
      self.desired_output = np.array(desired_output)

  def add_output_neuron(self, number=1):
    for i in range(number):
      self.output_neuron = np.append(self.output_neuron, Neuron(self.input_layer_size))
    self.output_layer_size += number

  def set_training_coeficient(self, coefficient):
    self.training_coefficient = coefficient
  
  def set_max_generations(self, max):
    self.max_generations = max

  def _activate_neuron(self, sample):
    output = np.empty(0)
    for neuron in self.output_neuron:
      for i in range(self.input_layer_size):
        neuron.input[i+1] = self.input_layer[sample][i]
        neuron.__calc__()
        output = np.append(output, neuron.output)
    return output
  
  def _adjust(self, neuron, output, sample):
    new_weight = neuron.weight + self.training_coefficient*(self.desired_output[sample]-output)*neuron.input
    neuron.set_weight(new_weight)

  def initiate(self, input):   
    if self.output_layer_size == 0:
      print('Erro: sem neurônios de saída')
      return

    print('Rede Iniciada!')
    input = np.array(input)
    self.output = np.empty(0)
    for sample in range(len(input)):
        self._activate_neuron(sample)
        output = np.empty(self.output_layer_size)
        for i in range(self.output_layer_size):
          output[i] = self.output_neuron[i].output
        self.output = np.append(self.output, output)

  def start_training(self):
    print('Treinamento Iniciado!')

    for gen in range(self.actual_generation, self.actual_generation+self.max_generations):
      if self.output_layer_size == 0:
        print('Erro: sem neurônios de saída')
        break
      print('Geração {}'.format(gen))
      erros = 0

      for sample in range(len(self.input_layer)):
        self._activate_neuron(sample)
        output = np.empty(self.output_layer_size)
        for i in range(self.output_layer_size):
          output[i] = self.output_neuron[i].output

        if output != self.desired_output[sample]:
          erros += 1
          self._adjust(self.output_neuron[i], output, sample)

      eficiency = (len(self.input_layer)-erros)/len(self.input_layer)
      print('Eficiência: {}%'.format(eficiency*100))
      if erros == 0: 
        print('Treinamento concluído com êxito!')
        break
      self.actual_generation += 1

    else:
      print('Treinamento interrompido: quantidade máxima de gerações atingida!')