################################################################
# architecture:
# 	input size: [3, 5, 15, 25]
# 	hidden dropout: [0.5, 0.6, 0.7, 0.8, 1]
# 	visible dropout: [0.8, 0.9, 0.95, 1]
# 	last fully connected layers: [1, 2, 3]
# 	increase filters: [0, 1]

# hyperparameter:
# 	learning rate: [1, 0.1, 0.01, 0.001]
# 	batch size: [16, 32, 64, 128]

# 	// changing learning rate over time is an option too; Like decreasing it after sometime.

# activation functions:
# 	activations: ['relu', 'selu', 'tanh', 'linear']
# 	// we can test 'elu' and 'leaky-relu' too.

# layers:
# 	pooling: ['MaxPooling', 'AveragePooling']

# ===========================

# selected:
# 	input size = [3, 5]
# 	hidden dropout: [0.5, 1]
# 	visible dropout: [0.8, 1]
# 	last fc layers: [1, 2]
#   architecture = [0, 1]
# 	learning rate: [0.1, 0.01]
# 	batch size: [16, 32, 64, 128]
# 	pooling: ['MaxPooling', 'AveragePooling']

# 384 cases
# split to 64 processes
# each one run 6 cases
################################################################

# parameters of models which we want to search
def create_parameters():
    parameters = []

    input_size = [3, 5]
    hidden_dropout = [0.5, 1]
    visible_dropout = [0.8, 1]
    NO_dense_layer = [2, 1]
    increase_filters = [0]
    architecture = [0, 1]

    learning_rate = [0.1, 0.01]
    batch_size = [16, 64, 128]
    pooling_type = ['MaxPooling', 'AveragePooling']

    for i0 in range(len(input_size)):
        for i1 in range(len(hidden_dropout)):
            for i2 in range(len(visible_dropout)):
                for i3 in range(len(NO_dense_layer)):
                    for i4 in range(len(increase_filters)):
                        # increase filters doesn't mean if input_size is 3. because there is only one block.
                        if (input_size[i0] == 3 and increase_filters[i4] == 1):
                            continue

                        for i8 in range(len(architecture)):
                            for i5 in range(len(learning_rate)):
                                for i6 in range(len(batch_size)):
                                    for i7 in range(len(pooling_type)):
                                        parameters.append(
                                            (input_size[i0],
                                            hidden_dropout[i1],
                                            visible_dropout[i2],
                                            NO_dense_layer[i3],
                                            increase_filters[i4],
                                            learning_rate[i5],
                                            batch_size[i6],
                                            pooling_type[i7],
                                            architecture[i8])
                                            )

    return parameters