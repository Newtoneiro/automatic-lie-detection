from collections import Counter


def get_class_distribution(labels):
    return dict(sorted(Counter((label.argmax().item() for label in labels)).items(), key=lambda i: i[0]))