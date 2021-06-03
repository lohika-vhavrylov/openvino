// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <string>
#include <iterator>
#include <samples/common.hpp>
#include <cstring>

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

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
cv::Mat imreadW(std::wstring input_image_path) {
    cv::Mat image;
    std::ifstream input_image_stream;
    input_image_stream.open(
        input_image_path.c_str(),
        std::iostream::binary | std::ios_base::ate | std::ios_base::in);
    if (input_image_stream.is_open()) {
        if (input_image_stream.good()) {
            input_image_stream.seekg(0, std::ios::end);
            std::size_t file_size = input_image_stream.tellg();
            input_image_stream.seekg(0, std::ios::beg);
            std::vector<char> buffer(0);
            std::copy(
                std::istreambuf_iterator<char>(input_image_stream),
                std::istreambuf_iterator<char>(),
                std::back_inserter(buffer));
            image = cv::imdecode(cv::Mat(1, file_size, CV_8UC1, &buffer[0]), cv::IMREAD_COLOR);
        } else {
            tcout << "Input file '" << input_image_path << "' processing error" << std::endl;
        }
        input_image_stream.close();
    } else {
        tcout << "Unable to read input file '" << input_image_path << "'" << std::endl;
    }
    return image;
}

std::string simpleConvert(const std::wstring & wstr) {
    std::string str;
    for (auto && wc : wstr)
        str += static_cast<char>(wc);
    return str;
}

int wmain(int argc, wchar_t *argv[]) {
#else

int main(int argc, char *argv[]) {
#endif
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 5) {
          tcout << "Usage : " << argv[0]
                << " <path_to_model> <path_to_image> <device_name> <iterations>"
                << std::endl;
          return EXIT_FAILURE;
        }

        const file_name_t input_model{argv[1]};
        const file_name_t input_image_path{argv[2]};
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        const std::string device_name = simpleConvert(argv[3]);
#else
        const std::string device_name{argv[3]};
#endif
        const int iterations{std::stoi(argv[4])};
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
        input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
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

        // --------------------------- 6. Prepare input --------------------------------------------------------
        using namespace std;
        /* Read input image to a blob and set it to an infer request without resize and layout conversions. */
        cv::Mat image = imread_t(input_image_path);
        cv::Mat f_image;
        image.convertTo(f_image, CV_32FC3, 1.);
        Blob::Ptr imgBlob = wrapFMat2Blob(f_image);
        // Blob::Ptr imgBlob = wrapMat2Blob(image);
        infer_request.SetBlob(input_name, imgBlob);  // infer_request accepts input blob of any size
        tcout << "In shape " << endl;
        for (auto dim : imgBlob->getTensorDesc().getDims()) {
            tcout << dim << " ";
        }
        tcout << "imgBlob->size() " << imgBlob->size() << " ";
        tcout << imgBlob->getTensorDesc().getPrecision() << " ";
        tcout << imgBlob->getTensorDesc().getLayout() << endl;

        // --------------------------- 7. Do inference --------------------------------------------------------
        auto start = std::chrono::high_resolution_clock::now();

        for (auto i{0}; i < iterations; ++i) {
          infer_request.StartAsync();
          infer_request.Wait(5000);
        }

        auto finish = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            finish - start)
                            .count();

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output ------------------------------------------------------
        Blob::Ptr output = infer_request.GetBlob(output_name);

        tcout << "Out shape " << endl;
        auto out_shape = output->getTensorDesc().getDims();
        for (auto dim : out_shape) {
          tcout << dim << " ";
        }
        tcout << "output->size() " << output->size() << " ";
        tcout << output->getTensorDesc().getPrecision() << " ";
        tcout << output->getTensorDesc().getLayout() << endl;

        tcout << "\nAverage inference time on " << iterations
              << " iterations: " << duration / (iterations * 1000) << " msec"
              << endl;

        // Print classification results
        ClassificationResult_t classificationResult(output, {input_image_path});
        classificationResult.print();
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool" << std::endl;
    return EXIT_SUCCESS;
}
