from collections import Counter


def get_class_distribution(labels):
    print("===> Class distribution <===")
    for label, count in sorted(Counter(labels).items(), key=lambda i: i[0]):
        print(f"{label}: {count}")
    print("=============><=============")
