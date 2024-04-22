package examples

import ml "../src"
import "core:fmt"

XOR_TRANING_DATA :: [12]f64{0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0}

Xor :: struct {
	a0:         ^ml.Matrix,
	w1, b1, a1: ^ml.Matrix,
	w2, b2, a2: ^ml.Matrix,
}

cost :: proc(using model: ^Xor, ti, to: ^ml.Matrix) -> (c: f64) {
	using ml
	assert(ti.rows == to.rows)
	assert(to.cols == a2.cols)

	x, y: ^Matrix
	d: f64
	for i in 0 ..< ti.rows {
		x = row_matrix(ti, i)
		y = row_matrix(to, i)
		copy_matrix(a0, x)
		forward_xor(model)
		for j in 0 ..< to.cols {
			d = at(a2, 0, j) - at(y, 0, j)
			c += d * d
		}
	}

	return c
}

setup :: proc(using m: ^Xor) {
	using ml

	a0 = new_matrix(1, 2)
	w1 = new_matrix(2, 2)
	b1 = new_matrix(1, 2)
	a1 = new_matrix(1, 2)
	w2 = new_matrix(2, 1)
	b2 = new_matrix(1, 1)
	a2 = new_matrix(1, 1)

	rand_matrix(w1, 0, 1)
	rand_matrix(b1, 0, 1)
	rand_matrix(w2, 0, 1)
	rand_matrix(b2, 0, 1)
}

forward_xor :: proc(using m: ^Xor) -> f64 {
	using ml

	dot(a1, a0, w1)
	sum_matrix(a1, b1)
	sigmoid_matrix(a1)

	dot(a2, a1, w2)
	sum_matrix(a2, b2)
	sigmoid_matrix(a2)

	return a2.es[0]
}
