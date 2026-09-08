def get_average(data):
    if len(data) == 0:
        return 0
    return sum(data) / len(data)

def get_std_deviation(data):
    if len(data) == 0:
        return 0
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

def get_min_max(data):
    if len(data) == 0:
        return (0, 0)
    return (min(data), max(data))

def get_95_confidence_interval(data):
    if len(data) == 0:
        return (0, 0)
    mean = sum(data) / len(data)
    std_dev = get_std_deviation(data)
    margin_of_error = 1.96 * (std_dev / (len(data) ** 0.5))
    return (mean - margin_of_error, mean + margin_of_error)