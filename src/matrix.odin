package ml

import "core:fmt"
import "core:math/rand"

Matrix :: struct {
	rows:   u64,
	cols:   u64,
	stride: u64,
	es:     []f64,
}

Row :: struct {
	cols:     u64,
	elements: []f64,
}

new_matrix :: proc(rows, cols: u64, allocator := context.allocator) -> (m: ^Matrix) {
	m = new(Matrix, allocator)
	m.rows = rows
	m.cols = cols
	m.stride = cols
	m.es = make(type_of(m.es), rows * cols, allocator)
	assert(m.es != nil)
	return
}

delete_matrix :: proc(m: ^Matrix) {
	delete(m.es)
	free(m)
}

copy_matrix :: proc(dst, src: ^Matrix) {
	assert(dst.rows == src.rows)
	assert(dst.cols == src.cols)
	for i in 0 ..< dst.rows {
		for j in 0 ..< dst.cols {
			ref_at(dst, i, j)^ = at(src, i, j)
		}
	}
}

at :: proc(m: ^Matrix, x, y: u64) -> f64 {
	assert(x < m.rows, fmt.tprintf("x: %v, y: %v, rows: %v", x, y, m.rows))
	assert(y < m.cols, fmt.tprintf("y: %v, cols: %v", y, m.cols))
	return m.es[x * m.stride + y]
}

ref_at :: proc(m: ^Matrix, x, y: u64) -> ^f64 {
	assert(x < m.rows, fmt.tprintf("x: %v, rows: %v", x, m.rows))
	assert(y < m.cols, fmt.tprintf("y: %v, cols: %v", y, m.cols))
	return &m.es[x * m.cols + y]
}

row_matrix :: proc(m: ^Matrix, row: u64) -> (r: ^Matrix) {
	r = new(Matrix)
	r.rows = 1
	r.cols = m.cols
	r.stride = m.stride
	r.es = m.es[row * m.cols:m.rows]
	return
}

fill_matrix :: proc(m: ^Matrix, value: f64) {
	for i in 0 ..< m.rows {
		for j in 0 ..< m.cols {
			ref_at(m, i, j)^ = value
		}
	}
}

dot :: proc(dst, a, b: ^Matrix) {
	assert(a.cols == b.rows)
	assert(dst.rows == a.rows)
	assert(dst.cols == b.cols)

	for i in 0 ..< dst.rows {
		for j in 0 ..< dst.cols {
			for k in 0 ..< a.cols {
				ref_at(dst, i, j)^ += at(a, i, k) * at(b, k, j)
			}
		}
	}
}

sum_matrix :: proc(dst, a: ^Matrix) {
	assert(dst.cols == a.cols)
	assert(dst.rows == a.rows)
	for x in 0 ..< dst.rows {
		for y in 0 ..< dst.cols {
			ref_at(dst, x, y)^ += at(a, x, y)
		}
	}
}

sigmoid_matrix :: proc(m: ^Matrix) {
	for x in 0 ..< m.rows {
		for y in 0 ..< m.cols {
			ref_at(m, x, y)^ = sigmoid(at(m, x, y))
		}
	}
}

rand_matrix :: proc(m: ^Matrix, min, max: f64, random := false) {
	r := rand.create(123)
	pr := &r

	if random do pr = nil

	for x in 0 ..< m.rows {
		for y in 0 ..< m.cols {
			ref_at(m, x, y)^ = rand.float64_range(min, max, pr)
		}
	}
}

print_matrix :: proc(m: ^Matrix, name := "_") {
	fmt.printf("%v = [\n", name)
	comma := true
	for x in 0 ..< m.rows {
		for y in 0 ..< m.cols {
			if y + 1 == m.cols do comma = false
			fmt.printf("  %2f%s", at(m, x, y), comma ? "," : "")
		}
		comma = true
		fmt.print("\n")
	}
	fmt.println("]")
}
