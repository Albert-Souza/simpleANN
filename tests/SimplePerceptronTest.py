from ..networks.simple_perceptron_network import Simple_Perceptron_Network

training_set = [[-0.6508, 0.1097, 4.0009], [-1.4492, 0.8896, 4.4005], [2.0850, 0.6876, 12.0710], [0.2626, 1.1476, 7.7985], [0.6418, 1.0234, 7.0427], [0.2569, 0.6730, 8.3265], [1.1155, 0.6043, 8.3265], [0.0914, 0.3399, 7.0677], [0.0121, 0.5256, 4.6316], [-0.0429, 0.4660, 5.4323], [0.4340, 0.6870, 8.2287], [0.2735, 1.0287, 7.1934], [0.4839, 0.4851, 7.4850], [0.4089, -0.1267, 5.5019], [1.4391, 0.1614, 8.5843], [-0.9115, -0.1973, 2.1962], [0.3654, 1.0475, 7.4858], [0.2144, 0.7515, 7.1699], [0.2013, 1.0014, 6.5489], [0.6483, 0.2183, 5.8991], [-0.1147, 0.2242, 7.2435], [-0.7970, 0.8795, 3.8762], [-1.0625, 0.6366, 2.4707], [0.5307, 0.1285, 5.6883], [-1.2200, 0.7777, 1.7252], [0.3957, 0.1076, 5.6623], [-0.1013, 0.5989, 7.1812], [2.4482, 0.9455, 11.2095], [2.0149, 0.6192, 10.9264], [0.2012, 0.2611, 5.4631]]

desired_output_set = [-1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1]

test_set = [[-0.3665, 0.0620, 5.9891], [-0.7842, 1.1267, 5.5912], [0.3012, 0.5611, 5.8234], [0.7757, 1.0648, 8.0677], [0.1570, 0.8028, 6.3040], [-0.7014, 1.0316, 3.6005], [0.3748, 0.1536, 6.1537], [-
0.6920, 0.9404, 4.4058], [-1.3970, 0.7141, 4.9262], [-1.8842, -0.2805, 1.2548]]

perceptron1 = Simple_Perceptron_Network(3)
print('Pesos iniciais: {}'.format(perceptron1.output_neuron[0].weight))
input('Pressione Enter para começar o treino 1')
perceptron1.set_inputs(training_set)
perceptron1.set_desired_output(desired_output_set)
perceptron1.start_training()
print('Pesos finais: {}'.format(perceptron1.output_neuron[0].weight))
perceptron1.initiate(test_set)
print('Resultado: {}'.format(perceptron1.output))

perceptron2 = Simple_Perceptron_Network(3)
print('Pesos iniciais: {}'.format(perceptron2.output_neuron[0].weight))
input('Pressione Enter para começar o treino 2')
perceptron2.set_inputs(training_set)
perceptron2.set_desired_output(desired_output_set)
perceptron2.start_training()
print('Pesos finais: {}'.format(perceptron2.output_neuron[0].weight))
perceptron2.initiate(test_set)
print('Resultado: {}'.format(perceptron2.output))

perceptron3 = Simple_Perceptron_Network(3)
print('Pesos iniciais: {}'.format(perceptron3.output_neuron[0].weight))
input('Pressione Enter para começar o treino 3')
perceptron3.set_inputs(training_set)
perceptron3.set_desired_output(desired_output_set)
perceptron3.start_training()
print('Pesos finais: {}'.format(perceptron3.output_neuron[0].weight))
perceptron3.initiate(test_set)
print('Resultado: {}'.format(perceptron3.output))

perceptron4 = Simple_Perceptron_Network(3)
print('Pesos iniciais: {}'.format(perceptron4.output_neuron[0].weight))
input('Pressione Enter para começar o treino 4')
perceptron4.set_inputs(training_set)
perceptron4.set_desired_output(desired_output_set)
perceptron4.start_training()
print('Pesos finais: {}'.format(perceptron4.output_neuron[0].weight))
perceptron4.initiate(test_set)
print('Resultado: {}'.format(perceptron4.output))

perceptron5 = Simple_Perceptron_Network(3)
print('Pesos iniciais: {}'.format(perceptron5.output_neuron[0].weight))
input('Pressione Enter para começar o treino 5')
perceptron5.set_inputs(training_set)
perceptron5.set_desired_output(desired_output_set)
perceptron5.start_training()
print('Pesos finais: {}'.format(perceptron5.output_neuron[0].weight))
perceptron5.initiate(test_set)
print('Resultado: {}'.format(perceptron5.output))
