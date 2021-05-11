// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>
#include <iterator>
#include <samples/common.hpp>

#include <inference_engine.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>

using namespace InferenceEngine;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#define tcout std::wcout
#define file_name_t std::wstring
#define imread_t imreadW
#define ClassificationResult_t ClassificationResultW
#else
#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult
#endif

int main(int argc, char *argv[]) {
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 4) {
            tcout << "Usage : " << argv[0] << " <path_to_model> <path_to_image> <device_name>" << std::endl;
            return EXIT_FAILURE;
        }

        const file_name_t input_model{argv[1]};
        const file_name_t input_image_path{argv[2]};
        const std::string device_name{argv[3]};
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load inference engine instance -------------------------------------
        Core ie;
        // -----------------------------------------------------------------------------------------------------

        // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) or ONNX (.onnx file) format
        CNNNetwork network = ie.ReadNetwork(input_model);

        if (network.getOutputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 output only");
        if (network.getInputsInfo().size() != 1) throw std::logic_error("Sample supports topologies with 1 input only");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
        std::string input_name = network.getInputsInfo().begin()->first;

        /* Mark input as resizable by setting of a resize algorithm.
         * In this case we will be able to set an input blob of any shape to an infer request.
         * Resize and layout conversions are executed automatically during inference */
        //input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_info->setLayout(Layout::NCHW);
        input_info->setPrecision(Precision::FP32);

        // --------------------------- Prepare output blobs ----------------------------------------------------
        DataPtr output_info = network.getOutputsInfo().begin()->second;
        std::string output_name = network.getOutputsInfo().begin()->first;

        output_info->setPrecision(Precision::FP32);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the device ------------------------------------------
        ExecutableNetwork executable_network = ie.LoadNetwork(network, device_name);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input
        float data[] = {-1.f, -2.f, +4.f, -5.f,  //
                        -7.f, +3.f, -8.f, -9.f,  //
                        -1.f, -2.f, -4.f, +5.f,  //
                        -7.f, -3.f, +8.f, -9.f,  //
                        //
                        -1.f, +2.f, -4.f, -5.f,  //
                        +7.f, -3.f, -8.f, +9.f,  //
                        -1.f, -2.f, +4.f, -5.f,  //
                        -7.f, +3.f, -8.f, -9.f,  //
                        //
                        -1.f, -2.f, -4.f, +5.f,  //
                        -7.f, -3.f, +8.f, -9.f,  //
                        -1.f, +2.f, -4.f, -5.f,  //
                        +7.f, -3.f, -8.f, +9.f};

        InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32,
                                          {1, 3, 4, 4},
                                          InferenceEngine::Layout::NCHW);

        Blob::Ptr imgBlob =
            InferenceEngine::make_shared_blob<float>(tDesc, data);

        const size_t in_size = imgBlob->size();
        auto in_dims = imgBlob->getTensorDesc().getDims();
        tcout << "Input elements "  // << in_size
              << std::endl;
        for (size_t i = 0; i < in_size; ++i) {
            tcout << (data[i] < 0 ? " " : "  ") << data[i];
            if ((i + 1) % in_dims[2] == 0 && i > 0) {
                tcout << std::endl;
                if ((i + 1) % (in_dims[2] * in_dims[3]) == 0 && i > 0) {
                    tcout << std::endl;
                }
            }
        }

        infer_request.SetBlob(
            input_name,
            imgBlob); // infer_request accepts input blob of any size
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        auto start =  std::chrono::high_resolution_clock::now();

        const auto iterations = 5000;
        for (auto i{0}; i < iterations; i++) {
          infer_request.Infer();
        }
        auto finish =  std::chrono::high_resolution_clock::now();

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);
        auto desc = output->getTensorDesc();
        auto dims = desc.getDims();
        // Expects dims to be of size 4

        size_t out_size = output->size();
        tcout << "Output elements "  // << out_size << ", expected " << in_size
              << std::endl;
        float *out_data = output.get()->buffer();
        for (size_t i = 0; i < out_size; ++i) {
          tcout << "  " << out_data[i];
          if ((i + 1) % dims[2] == 0 && i > 0) {
            tcout << std::endl;
            if ((i + 1) % (dims[2] * dims[3]) == 0) {
              tcout << std::endl;
            }
          }
        }

        std::cout << "\nAvg execution time (per " << iterations << " iterations): "
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                         finish - start)
                             .count() /
                         (iterations)
                  << " microseconds" << std::endl;
    } catch (const std::exception &ex) {
      std::cerr << ex.what() << std::endl;
      return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
