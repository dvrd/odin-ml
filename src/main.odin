package ml

import "core:fmt"
import "core:math"
import "core:testing"

@(test)
test_matrix_sum :: proc(t: ^testing.T) {
	using testing

	a := new_matrix(2, 2)
	b := new_matrix(2, 2)

	fill_matrix(a, 1)
	fill_matrix(b, 1)

	sum_matrix(a, b)
	expected := []f64{2, 2, 2, 2}
	for val, idx in expected {
		expect(t, a.es[idx] == val, fmt.tprintf("sum_matrix: expected %f, got %f", val, a.es[idx]))
	}
}

@(test)
test_matrix_dot_product :: proc(t: ^testing.T) {
	using testing

	a := new_matrix(1, 2)
	rand_matrix(a, 5, 10)

	b := new_matrix(2, 2)
	fill_matrix(b, 1)

	dst := new_matrix(1, 2)

	dot(dst, a, b)
	expected := []f64{11.450617333721468, 11.450617333721468}
	for val, idx in expected {
		expect(
			t,
			dst.es[idx] == val,
			fmt.tprintf("sum_matrix: expected %v, got %v", val, dst.es[idx]),
		)
	}
}

@(test)
test_get_matrix_row :: proc(t: ^testing.T) {
	using testing
	a := new_matrix(2, 2)
	rand_matrix(a, 5, 10)
	r := row_matrix(a, 0)
	expected := []f64{5.864215696240108, 5.5864016374813605}
	for val, idx in expected {
		expect(
			t,
			r.es[idx] == val,
			fmt.tprintf("get_matrix_row: expected %v, got %v", val, r.es[idx]),
		)
	}
}

main :: proc() {}
