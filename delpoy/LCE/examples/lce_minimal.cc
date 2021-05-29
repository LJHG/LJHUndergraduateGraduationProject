#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This file is based on the TF lite minimal example where the
// "BuiltinOpResolver" is modified to include the "Larq Compute Engine" custom
// ops. Here we read a binary model from disk and perform inference by using the
// C++ interface. See the BUILD file in this directory to see an example of
// linking "Larq Compute Engine" cutoms ops to your inference binary.

using namespace tflite;

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 2) {
    fprintf(stderr, "lce_minimal <tflite model>\n");
    return 1;
  }
  const char* filename = argv[1];

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  // printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  float* input = interpreter->typed_input_tensor<float>(0);
  // suppose the input size is 224*224*3
  // so lets set them all to 1,2,3
  // 224 * 224 = 50176
  for(int i=0;i<50176;i++)
  {
    input[i*3 + 0] = 1;
    input[i*3 + 1] = 2;
    input[i*3 + 2] = 3;
  }
  std::cout<<"all inputs have been set to (1,2,3) format"<<"\n";


  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // printf("\n\n=== Post-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  float* output = interpreter->typed_output_tensor<float>(0);
  // TODO(user): Insert getting data out code.

  std::cout<<"show the first 100 outputs"<<"\n";
  for(int i=0;i<20;i++){
    std::cout<<output[i]<<" ";
  }

  std::cout<<"\n 打印前五大的index"<<"\n";
  std::vector<std::pair<float,int>> ans;
  for(int i=0;i<1000;i++){
    ans.push_back(std::make_pair(output[i]*-1,i));
  }
  std::sort(ans.begin(),ans.end());

  for(int i=0;i<20;i++){
    auto x = ans[i];
    std::cout<<x.second<<" ";
  }

  printf("\n");
  printf("hello world \n");

  return 0;
}
