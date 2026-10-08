"""Matrix-based privacy-aware aggregation scheme."""

import numpy as np #type: ignore


def mod_inv(value, modulus):
    return pow(int(value), -1, modulus)


def mat_mod_inv(matrix, modulus):
    matrix = np.array(matrix, dtype=int) % modulus
    size = matrix.shape[0]
    augmented = np.concatenate([matrix, np.eye(size, dtype=int)], axis=1) % modulus
    for column in range(size):
        pivot = next((row for row in range(column, size) if augmented[row, column] % modulus != 0), None)
        if pivot is None:
            raise ValueError("Matrix not invertible modulo p")
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] = (augmented[column] * mod_inv(augmented[column, column], modulus)) % modulus
        for row in range(size):
            if row != column:
                factor = augmented[row, column] % modulus
                augmented[row] = (augmented[row] - factor * augmented[column]) % modulus
    return augmented[:, size:] % modulus


def random_invertible_matrix(size, modulus, rng):
    while True:
        matrix = rng.integers(0, modulus, size=(size, size), dtype=np.int64)
        try:
            mat_mod_inv(matrix, modulus)
            return matrix % modulus
        except Exception:
            continue


def split_vector_by_S(vector, split_mask, modulus, rng):
    vector = np.array(vector, dtype=int) % modulus
    first = np.zeros_like(vector)
    second = np.zeros_like(vector)
    for index in range(len(vector)):
        if split_mask[index] == 1:
            first[index] = second[index] = vector[index]
        else:
            random_value = int(rng.integers(0, modulus))
            first[index] = random_value
            second[index] = (vector[index] - random_value) % modulus
    return first % modulus, second % modulus


def split_selector_by_S(vector, split_mask, modulus, rng):
    vector = np.array(vector, dtype=int) % modulus
    first = np.zeros_like(vector)
    second = np.zeros_like(vector)
    for index in range(len(vector)):
        if split_mask[index] == 0:
            first[index] = second[index] = vector[index]
        else:
            random_value = int(rng.integers(0, modulus))
            first[index] = random_value
            second[index] = (vector[index] - random_value) % modulus
    return first % modulus, second % modulus


class PrivacyAwareAggregation:
    def __init__(self, n, p=2147483647, seed=42):
        self.n, self.p = n, p
        self.rng = np.random.default_rng(seed)
        self.S = self.rng.integers(0, 2, size=n, dtype=np.int64)
        self.X1 = random_invertible_matrix(n, p, self.rng)
        self.X2 = random_invertible_matrix(n, p, self.rng)
        self.Y1 = random_invertible_matrix(n, p, self.rng)
        self.Y2 = random_invertible_matrix(n, p, self.rng)
        self.Y3 = random_invertible_matrix(n, p, self.rng)
        self.Y4 = random_invertible_matrix(n, p, self.rng)
        self.X1_inv, self.X2_inv = mat_mod_inv(self.X1, p), mat_mod_inv(self.X2, p)
        self.Y1_inv, self.Y2_inv = mat_mod_inv(self.Y1, p), mat_mod_inv(self.Y2, p)
        self.Y3_inv, self.Y4_inv = mat_mod_inv(self.Y3, p), mat_mod_inv(self.Y4, p)

    def generate_user_keys(self, user_id):
        first = self.rng.integers(0, self.p, size=(self.n, self.n), dtype=np.int64)
        third = self.rng.integers(0, self.p, size=(self.n, self.n), dtype=np.int64)
        return {"user_id": user_id, "S": self.S.copy(), "X1p": first % self.p,
                "X1pp": (self.X1_inv - first) % self.p, "X2p": third % self.p,
                "X2pp": (self.X2_inv - third) % self.p}

    def encrypt_reading(self, reading, user_keys):
        first, second = split_vector_by_S(reading, self.S, self.p, self.rng)
        c1 = (self.Y1_inv @ (user_keys["X1p"] @ first)) % self.p
        c2 = (self.Y2_inv @ (user_keys["X1pp"] @ first)) % self.p
        c3 = (self.Y3_inv @ (user_keys["X2p"] @ second)) % self.p
        c4 = (self.Y4_inv @ (user_keys["X2pp"] @ second)) % self.p
        return np.concatenate([c1, c2, c3, c4]) % self.p

    def aggregate_ciphertexts(self, ciphertexts):
        return np.sum(np.stack(ciphertexts, axis=0), axis=0) % self.p

    def decrypt_full(self, aggregate):
        aggregate = np.array(aggregate, dtype=int) % self.p
        n = self.n
        q_first = (self.X1 @ ((self.Y1 @ aggregate[:n] + self.Y2 @ aggregate[n:2*n]) % self.p)) % self.p
        q_second = (self.X2 @ ((self.Y3 @ aggregate[2*n:3*n] + self.Y4 @ aggregate[3*n:4*n]) % self.p)) % self.p
        output = np.zeros(n, dtype=np.int64)
        for index in range(n):
            output[index] = q_first[index] % self.p if self.S[index] == 1 else (q_first[index] + q_second[index]) % self.p
        return output

    def encrypt_selector(self, index):
        selector = np.zeros(self.n, dtype=np.int64)
        selector[index] = 1
        first, second = split_selector_by_S(selector, self.S, self.p, self.rng)
        return np.concatenate([(first @ self.X1 @ self.Y1) % self.p,
                               (first @ self.X1 @ self.Y2) % self.p,
                               (second @ self.X2 @ self.Y3) % self.p,
                               (second @ self.X2 @ self.Y4) % self.p]) % self.p

    def decrypt_partial(self, aggregate, index):
        selector = self.encrypt_selector(index)
        return int(np.dot(aggregate % self.p, selector % self.p) % self.p)
