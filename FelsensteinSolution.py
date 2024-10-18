import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_in_tree_data():
    tree_data = np.genfromtxt('tree_data.txt', delimiter=' ', dtype=float)
    return tree_data


def find_number_of_leaves_and_nodes_underneath(tree_data=None, node_query_label=19):
    if tree_data is None:
        tree_data = read_in_tree_data()

    node_labels = tree_data.T[[0, 1, 3]]
    node_labels = node_labels.T.astype(int)
    node_labels_1d = node_labels.flatten()
    # We want to keep the tree distances as floats
    tree_distances = tree_data.T[[2, 4]]
    tree_distances = tree_distances.T
    tree_distances_1d = tree_distances.flatten()

    matrix_indicies = np.where(node_labels_1d == node_query_label)[0]

    matrix_indicies_mod = matrix_indicies + np.repeat(3, len(matrix_indicies))
    matrix_indicies_mod = matrix_indicies_mod % 3

    # This is either an empty array, or an array of length 1 containing the index in matrix_indicies_mod which
    # contains the query node in the first column
    start_pos = np.where(matrix_indicies_mod == 0)[0]

    if len(start_pos) == 0:
        # If the node is not in the first column, then it contains no internal nodes, or leaves below it
        return 0, 0
    # These are the node labels in the columns immediately after the first column
    number_of_position_columns = node_labels.shape[1]
    # We don't want the index of the query node itself, just of the ones directly underneath it
    position_indicies_of_nodes_immediately_under = np.arange(0, number_of_position_columns)[1:] + matrix_indicies[
        start_pos]

    distance_indicies_of_nodes_immediately_under = position_indicies_of_nodes_immediately_under - (
            np.floor((position_indicies_of_nodes_immediately_under / 3)) + 1)
    distance_indicies_of_nodes_immediately_under = distance_indicies_of_nodes_immediately_under.astype(int)

    nodes_immediately_under = node_labels_1d[position_indicies_of_nodes_immediately_under]
    distances_immediately_under = tree_distances_1d[distance_indicies_of_nodes_immediately_under]
    # Now, we want to see if any of the nodes immediately under have any other nodes under them, and hence determine
    # whether they are a leaf or an internal node
    internal_nodes_total = 0
    leaves_total = 0

    leaf_distances_from_root = []

    for i, sub_node in enumerate(nodes_immediately_under):
        # The node only has nodes immediately under if it is in the first column, so follow the same logic as above
        matrix_indicies_sub = np.where(node_labels_1d == sub_node)[0]

        matrix_indicies_mod_sub = matrix_indicies_sub + np.repeat(3, len(matrix_indicies_sub))
        matrix_indicies_mod_sub = matrix_indicies_mod_sub % 3

        # This is either an empty array, or an array of length 1 containing the index in matrix_indicies_mod which
        # contains the query node in the first column
        start_pos_sub = np.where(matrix_indicies_mod_sub == 0)[0]

        if len(start_pos_sub) == 0:
            # If the sub node is not in the first column, then it is a leaf
            leaves_total += 1
            leaf_distances_from_root.append(distances_immediately_under[i])
        else:
            # If it is in the first column, then it is an internal node with leaves below it
            internal_nodes_total += 1
            leaves_total += 2

            # We need to get the distances of those leaves below it from that internal node, and then add them
            # to the distance of the internal node, to get the distance of each leaf from the root.
            position_indicies_of_nodes_immediately_under_sub = np.arange(0, number_of_position_columns)[1:] + \
                                                               matrix_indicies_sub[
                                                                   start_pos_sub]

            distance_indicies_of_nodes_immediately_under_sub = position_indicies_of_nodes_immediately_under_sub - (
                    np.floor((position_indicies_of_nodes_immediately_under_sub / 3)) + 1)
            distance_indicies_of_nodes_immediately_under_sub = distance_indicies_of_nodes_immediately_under_sub.astype(
                int)

            nodes_immediately_under_sub = node_labels_1d[position_indicies_of_nodes_immediately_under_sub]
            distances_immediately_under_sub = tree_distances_1d[distance_indicies_of_nodes_immediately_under_sub]
            distances_immediately_under_sub = distances_immediately_under_sub + \
                                              distances_immediately_under[i]

            leaf_distances_from_root.extend(distances_immediately_under_sub)

    return internal_nodes_total, leaves_total, leaf_distances_from_root


def create_table_with_totals():
    tree_data = read_in_tree_data()
    nodes_to_try = tree_data[:, 0]
    leaves_total_array = np.zeros(len(nodes_to_try))
    internal_nodes_total_array = np.zeros(len(nodes_to_try))

    for i, node in enumerate(nodes_to_try):
        internal_nodes_total, leaves_total, leaf_distances_from_root = find_number_of_leaves_and_nodes_underneath(
            tree_data, node)
        internal_nodes_total_array[i] = internal_nodes_total
        leaves_total_array[i] = leaves_total
    totals_table = np.column_stack((internal_nodes_total_array, leaves_total_array))

    table_with_totals = np.hstack((tree_data, totals_table))
    return table_with_totals


def make_jc_transition_matrix(t, mu=0.25):
    transition_matrix = np.zeros((4, 4))

    off_diagonal_elements = 0.25 * (1 - np.exp((-4 * mu * t)))
    diagonal_elements = 0.25 * (1 + 3 * np.exp(-4 * mu * t))

    transition_matrix = transition_matrix + off_diagonal_elements
    np.fill_diagonal(transition_matrix, diagonal_elements)

    return transition_matrix


def determine_leaves():
    # Only nodes in the last two node label columns can be leaves

    tree_data = read_in_tree_data()

    node_labels = tree_data.T[[0, 1, 3]]
    node_labels = node_labels.T.astype(int)

    last_two_column_node_labels = node_labels[:, [-2, -1]].T.flatten()

    # Now, just run them through the function and keep the ones that have no internal nodes or leaves below them
    leaves = []
    for node_label in last_two_column_node_labels:
        underneath = find_number_of_leaves_and_nodes_underneath(tree_data, node_label)
        # The function returns more than two elements, containing the leaf distances from root if the node in question
        # has leaves underneath, which it will if it is not a leaf itself
        if len(underneath) == 2:
            leaves.append(node_label)

    return leaves


def calculate_log_likelihoods(leaf_values=np.array(['A', 'C', 'A', 'G', 'G', 'A', 'T', 'C', 'A', 'T']),
                              mutation_rates=np.linspace(0.001, 1)):
    initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])
    print("Using mutation rates: ", mutation_rates)

    leaf_likelihoods = np.zeros((10, 1, 4))
    leaf_likelihoods[:, :, 0][np.where(leaf_values == 'A')[0]] = 1
    leaf_likelihoods[:, :, 1][np.where(leaf_values == 'T')[0]] = 1
    leaf_likelihoods[:, :, 2][np.where(leaf_values == 'C')[0]] = 1
    leaf_likelihoods[:, :, 3][np.where(leaf_values == 'G')[0]] = 1

    tree_data = read_in_tree_data()

    node_labels = tree_data.T[[0, 1, 3]]
    node_labels = node_labels.T.astype(int)
    node_labels_1d = node_labels.flatten()
    node_labels_last_two_cols = node_labels[:, [-2, -1]]
    # We want to keep the tree distances as floats
    tree_distances = tree_data.T[[2, 4]]
    tree_distances = tree_distances.T
    tree_distances_1d = tree_distances.flatten()

    extra_data_table = create_table_with_totals()
    # print(extra_data_table)
    internal_nodes = extra_data_table[:, -2]
    number_of_node_labels = len(np.unique(node_labels_1d))

    log_likelihood_history = np.zeros(len(mutation_rates))
    for i, mutation_rate in enumerate(mutation_rates):
        log_likelihood = np.zeros((number_of_node_labels, 1, 4))
        log_likelihood[0:10] = leaf_likelihoods
        node = 0
        # The recursion counter keeps track of how many times we need to go down to the tips again
        recursion_counter = 0
        index_in_table = np.where(internal_nodes == 0)[0]
        if len(index_in_table) > 0:
            # There may be multiple nodes with only leaves as direct descendants - we start by picking the first one
            # in the table
            index_in_table = np.array([index_in_table[recursion_counter]])
        while len(index_in_table) != 0:
            index_in_table = index_in_table[0]
            node_labels_row = node_labels[index_in_table].flatten()
            node = node_labels_row[0]
            child_1 = node_labels_row[1]
            child_2 = node_labels_row[2]

            node_index = node - 1
            child_1_index = child_1 - 1
            child_2_index = child_2 - 1

            tree_distances_row = tree_distances[index_in_table].flatten()
            distance_1 = tree_distances_row[0]
            distance_2 = tree_distances_row[1]

            transition_matrix_1 = make_jc_transition_matrix(distance_1, mutation_rate)
            transition_matrix_2 = make_jc_transition_matrix(distance_2, mutation_rate)

            log_likelihoods_1 = log_likelihood[child_1_index]
            log_likelihoods_2 = log_likelihood[child_2_index]

            # We need to have already worked out the child log likelihoods - if we haven't, we need to work up to them
            uninitialised_children_bool = np.all(log_likelihoods_1 == 0) + np.all(log_likelihoods_2 == 0)
            if uninitialised_children_bool >= 1:
                recursion_counter += 1
                index_in_table = np.array([np.where(internal_nodes == 0)[0][recursion_counter]])
                continue

            parent_log_likelihood_vector = (np.sum((transition_matrix_1 * log_likelihoods_1), axis=1)) * (np.sum(
                (transition_matrix_2 * log_likelihoods_2), axis=1))
            log_likelihood[node_index] = parent_log_likelihood_vector

            index_in_table = np.where(node_labels_last_two_cols == node)[0]

        # Then we have reached the root
        tree_likelihood = initial_distribution * log_likelihood[node_index]
        tree_likelihood = np.sum(tree_likelihood)
        log_likelihood_history[i] = tree_likelihood

    log_likelihood_history = np.log(log_likelihood_history)
    print("Max likelihood value of mu = ", mutation_rates[np.argmax(log_likelihood_history)])
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(mutation_rates, log_likelihood_history)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\ln{L}$")
    fig.savefig("log_likelihood_vs_mutation_rate.png")

    return log_likelihood_history


def mle_with_randomly_assigned_leaf_values(number_of_iterations=1000):
    mutation_rates = np.linspace(0.01, 1, 100)
    possible_leaf_values = np.array(['A', 'T', 'C', 'G'])

    mle_history = np.zeros(number_of_iterations)
    for i in range(0, number_of_iterations):
        random_leaf_values = np.random.choice(possible_leaf_values, 10)
        print(random_leaf_values)
        log_likelihood_history = calculate_log_likelihoods(random_leaf_values, mutation_rates)
        mle_history[i] = mutation_rates[np.argmax(log_likelihood_history)]

    fig = plt.figure()
    ax = plt.axes()
    ax.hist(mle_history)
    ax.set_xlabel(r"$\ln{L}$")
    ax.set_ylabel("Frequency")
    fig.savefig("mle_hist.png")

    return mle_history


calculate_log_likelihoods()
mle_with_randomly_assigned_leaf_values()

table = pd.DataFrame(create_table_with_totals())
table[[0, 1, 3, 5, 6]] = table[[0, 1, 3, 5, 6]].astype(int)

print(table)

table.to_csv("TreeTable.csv", index=False, header=False)
