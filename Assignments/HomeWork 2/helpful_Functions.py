import numpy

lambda_value = 0


def get_centroid_vector(set_of_vectors, indices_of_vectors):
    """This Function calculates the centroid for given vector"""
    len_of_vectors = len(indices_of_vectors)
    size_of_individual_vector = len(set_of_vectors[indices_of_vectors[0]])
    centroid_vector = []
    index = 0
    while index < size_of_individual_vector:
        centroid_vector.append(__get_centroid_for_particular_index(len_of_vectors,
                                                                   set_of_vectors, index, indices_of_vectors))
        index += 1
    return centroid_vector


def __get_centroid_for_particular_index(len_of_vectors, set_of_vectors, index, indices_of_vectors):
    """This Function calculates a particular centroid of a given vector set"""
    i = 0
    sum_val = 0.0
    while i < len_of_vectors:
        sum_val += set_of_vectors[indices_of_vectors[i]][index]
        i += 1
    return sum_val/len_of_vectors


def get_difference_in_distance(first_vector, second_vector):
    """This Function calculates the distance between two vectors"""
    dist = numpy.linalg.norm(numpy.array(first_vector) - numpy.array(second_vector))
    return dist


def __internal_calculate_max_r(set_of_vectors, indices_of_vectors, corresponding_m_vector):
    """This Function calculates the max value of r for a given vector set"""
    index = 1
    max_distance = get_difference_in_distance(set_of_vectors[indices_of_vectors[0]], corresponding_m_vector)
    while index < len(indices_of_vectors):
        temp_distance = get_difference_in_distance(set_of_vectors[indices_of_vectors[index]], corresponding_m_vector)
        if temp_distance > max_distance:
            max_distance = temp_distance
        index += 1
    return max_distance


def calculate_lambda(m_positive, m_negative, positive_indices, negative_indices, vector_set):
    """This Function calculates the lambda value for two given sets"""
    r = get_difference_in_distance(m_positive, m_negative)
    r_positive = __internal_calculate_max_r(vector_set, positive_indices, m_positive)
    r_negative = __internal_calculate_max_r(vector_set, negative_indices, m_negative)
    max_lambda = r/(r_positive+r_negative)
    global lambda_value  # lambda_value is stored to be used later for reduced set calculation
    lambda_value = max_lambda
    return lambda_value


def calculate_reduced_input_data_set(vector_set, positive_indices, negative_indices, m_positive, m_negative):
    """This Function calculates the reduced convex hulls for two given sets"""
    reduced_data_set = vector_set[:]
    index = 0
    while index < len(positive_indices):
        reduced_data_set[positive_indices[index]] = __internal_calculation_for_reduced_data_set(
            vector_set[positive_indices[index]], m_positive)
        index += 1
    index = 0
    while index < len(negative_indices):
        reduced_data_set[negative_indices[index]] = __internal_calculation_for_reduced_data_set(
            vector_set[negative_indices[index]], m_negative)
        index += 1
    return numpy.array(reduced_data_set)


def __internal_calculation_for_reduced_data_set(vector, corresponding_centroid_vector):
    """This Function calculates the reduced form of a given vector. Lambda value is used from previous calculation"""
    global lambda_value
    x_prime_vector = []
    index = 0
    while index < len(vector):
        temp = (lambda_value * vector[index]) + ((1.0 - lambda_value) * corresponding_centroid_vector[index])
        x_prime_vector.append(temp)
        index += 1
    return numpy.array(x_prime_vector)
