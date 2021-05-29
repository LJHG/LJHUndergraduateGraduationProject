#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <jni.h>
#include <string>

#include "larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"


//assets manager
// #include <android/asset_manager.h> 
// #include <android/asset_manager_jni.h> 
#include <android/log.h> 
#define TAG "HELLO" 
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__) 
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG , TAG, __VA_ARGS__) 
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO , TAG, __VA_ARGS__) 
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN , TAG, __VA_ARGS__) 
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR , TAG, __VA_ARGS__) 



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



extern "C" JNIEXPORT jstring JNICALL
Java_com_ljh_bnndemo_Net_testForJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from lce";
    std::cout<<"hello there!!!!!!!!!";
    return env->NewStringUTF(hello.c_str());
}

//use a interpreter as a global variable
std::unique_ptr<Interpreter> interpreter;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_ljh_bnndemo_Net_loadModel(
        JNIEnv* env,
        jobject thiz,
        jobject model_buffer) {

  char* buffer = static_cast<char*>(env->GetDirectBufferAddress(model_buffer));
  size_t buffer_size = static_cast<size_t>(env->GetDirectBufferCapacity(model_buffer));

  // Load model
  // std::unique_ptr<tflite::FlatBufferModel> model =
  //     tflite::FlatBufferModel::BuildFromFile("quicknet.tflite");

  // use build from buffer
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromBuffer(buffer,buffer_size);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter
  tflite::ops::builtin::BuiltinOpResolver resolver;
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  InterpreterBuilder builder(*model, resolver);
  //std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  // printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  LOGI("model load succeed!!!");
    
   return true;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_ljh_bnndemo_Net_predict(
        JNIEnv* env,
        jobject thiz,
        jfloatArray arr) {

    float *jInput;
    jInput = env->GetFloatArrayElements(arr, 0);
    const jint length = env->GetArrayLength(arr);

  LOGI(".................start to predict....................");
    //   // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  float* input = interpreter->typed_input_tensor<float>(0);

  //直接赋地址会出现奇怪的结果，所以下面改成循环赋值了
  //input = jInput;
  // suppose the input size is 32*32*3
  // 32*32 = 1024
  for(int i=0;i<1024;i++)
  {
    input[i*3 + 0] = jInput[i*3 + 0];
    input[i*3 + 1] = jInput[i*3 + 1];
    input[i*3 + 2] = jInput[i*3 + 2];
  }
  // for(int i=0;i<20;i++){
  //   LOGI(std::to_string(int(input[i])).c_str());LOGI(" ");
  // }
  // LOGI("\n");
  // LOGI("all inputs have been set to (1,2,3) format \n");


  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);

  // printf("\n\n=== Post-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  float* output = interpreter->typed_output_tensor<float>(0);
  // TODO(user): Insert getting data out code.

  // LOGI("show the first 100 outputs \n");
  // for(int i=0;i<20;i++){
  //   std::cout<<output[i]<<" ";
  // }

  //LOGI("\n 打印前五大的index \n");
  //使用pair排序后获得从大到小的index
  // std::vector<std::pair<float,int>> ans;
  // for(int i=0;i<1000;i++){
  //   ans.push_back(std::make_pair(output[i]*-1,i));
  // }
  // std::sort(ans.begin(),ans.end());

  // for(int i=0;i<20;i++){
  //   auto x = ans[i];
  //   LOGI(std::to_string(x.second).c_str());LOGI(" ");
  // }

  //输出
  //CIFAR100对应100分类
  float *log_mel = new float[100];
  for(int i=0;i<100;i++){
    log_mel[i] = output[i];
  }
  jfloatArray result = env->NewFloatArray(100);
  env -> SetFloatArrayRegion(result,0,100,log_mel);

  LOGI("predict over \n");
    
  return result;
}


