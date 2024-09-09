// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=abs --input="2xf32=-2 3") | FileCheck %s
// RUN: (iree-compile --iree-hal-target-backends=llvm-cpu %s | iree-run-module --device=local-task --module=- --function=abs --input="2xf32=-2 3") | FileCheck %s

// CHECK-LABEL: EXEC @abs
func.func @abs(%lhs : tensor<1x1024x2048xf32>, %rhs : tensor<1x2048x512xf32>) -> (tensor<1x1024x512xf32>) {
  %result = "tosa.matmul"(%lhs, %rhs) : (tensor<1x1024x2048xf32>, tensor<1x2048x512xf32>) -> (tensor<1x1024x512xf32>)
  return %result : tensor<1x1024x512xf32>
}
  // INPUT-BUFFERS: result[1]: hal.buffer_view
  // INPUT-BUFFERS-NEXT: 2xf32=-2.0 3.0