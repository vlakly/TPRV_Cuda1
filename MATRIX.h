#pragma once

void MAT_fill_empty(uint32_t* m, int size) {
	for (int i = 0; i < size * size; i++) {
		m[i] = 0;
	}
}
void MAT_fill_random(uint32_t* m, int size) {
	for (int i = 0; i < size * size; i++) {
		m[i] = rand() % 4;
	}
}
void MAT_print(uint32_t* m, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			std::cout << m[j + i * size] << " ";
		}
		std::cout << "\n";
	}
}
void MAT_scalar_multiply(uint32_t* res, uint32_t* mA, uint32_t* mB, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				res[j + i * size] += mA[k + i * size] * mB[j + k * size];
			}
		}
	}
}
bool MAT_check_equality(uint32_t* mA, uint32_t* mB, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (mA[j + i * size] != mB[j + i * size]) {
				return false;
			}
		}
	}
	return true;
}