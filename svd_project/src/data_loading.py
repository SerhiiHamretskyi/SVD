import numpy as np
import os

def load_movielens_100k(path: str) -> np.ndarray:

    num_users = 943
    num_items = 1682

    R = np.zeros((num_users, num_items))

    with open(path, "r") as f:
        for line in f:
            user, item, rating, _ = line.strip().split("\t")

            user = int(user) - 1
            item = int(item) - 1
            rating = float(rating)

            R[user, item] = rating

    return R