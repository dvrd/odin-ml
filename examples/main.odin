package examples

import ml "../src"
import "core:fmt"

main :: proc() {
	model: Xor
	trainig_data := XOR_TRANING_DATA
	stride: u64 = 3
	input := ml.Matrix {
		rows   = cast(u64)len(trainig_data) / stride,
		cols   = 2,
		stride = stride,
		es     = trainig_data[:],
	}
	out := ml.Matrix {
		rows   = cast(u64)len(trainig_data) / stride,
		cols   = 1,
		stride = stride,
		es     = trainig_data[2:],
	}

	ml.print_matrix(&input, "in")
	ml.print_matrix(&out, "out")

	setup(&model)
	fmt.printfln("cost: %f", cost(&model, &input, &out))
	for i in 0 ..< 2 {
		for j in 0 ..< 2 {
			ml.ref_at(model.a0, 0, 0)^ = cast(f64)i
			ml.ref_at(model.a0, 0, 1)^ = cast(f64)j
			y := forward_xor(&model)
			fmt.printfln("XOR(%d, %d) = %f", i, j, y)
		}
	}
}
