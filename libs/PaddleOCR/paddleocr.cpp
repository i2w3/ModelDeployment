


// std::vector<TextDetOutput> TextDetector::infer(const cv::Mat& input_image){
// #ifdef IS_DEBUG
//     std::chrono::steady_clock::time_point start, end;
//     std::chrono::duration<double, std::milli> preprocess_time;
//     start = std::chrono::high_resolution_clock::now();
// #endif
//     // 前处理
//     cv::Mat blob = this->preprocess(input_image);
//     this->net.setInput(blob);
// #ifdef IS_DEBUG
//     end = std::chrono::high_resolution_clock::now();
//     preprocess_time = end - start;
//     std::cout << "Preprocessing time: " << preprocess_time.count() << " ms" << std::endl;
//     start = std::chrono::high_resolution_clock::now();
// #endif
//     // 正向传播
//     std::vector<cv::Mat> outputs;
//     this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());
// #ifdef IS_DEBUG
//     end = std::chrono::high_resolution_clock::now();
//     preprocess_time = end - start;
//     std::cout << "model forward time: " << preprocess_time.count() << " ms" << std::endl;
//     start = std::chrono::high_resolution_clock::now();
// #endif
//     // 后处理
//     auto result = this->postProcess(outputs);
// #ifdef IS_DEBUG
//     end = std::chrono::high_resolution_clock::now();
//     preprocess_time = end - start;
//     std::cout << "Postprocessing time: " << preprocess_time.count() << " ms" << std::endl;
// #endif
//     return result;
// }


// TextClassifier::TextClassifier(const std::string& modelPath) 
//     : Model(modelPath, 48, 192) {
//     std::cout << "Initialized TextClassifier with model path: " << modelPath << std::endl;
//     std::cout << "Input size set to " << this->height << "x" << this->width << std::endl;
// }

// AngleType TextClassifier::infer(const cv::Mat& input_image) {
// #ifdef IS_DEBUG
//     std::chrono::steady_clock::time_point start, end;
//     std::chrono::duration<double, std::milli> preprocess_time;
//     start = std::chrono::high_resolution_clock::now();
// #endif
//     // 前处理
//     cv::Mat blob = this->preprocess(input_image);
//     this->net.setInput(blob);
// #ifdef IS_DEBUG
//     end = std::chrono::high_resolution_clock::now();
//     preprocess_time = end - start;
//     std::cout << "Preprocessing time: " << preprocess_time.count() << " ms" << std::endl;
//     start = std::chrono::high_resolution_clock::now();
// #endif
//     // 正向传播
//     std::vector<cv::Mat> outputs;
//     this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());
// #ifdef IS_DEBUG
//     end = std::chrono::high_resolution_clock::now();
//     preprocess_time = end - start;
//     std::cout << "model forward time: " << preprocess_time.count() << " ms" << std::endl;
//     start = std::chrono::high_resolution_clock::now();
// #endif
//     // 后处理
//     auto result = this->postProcess(outputs);
// #ifdef IS_DEBUG
//     end = std::chrono::high_resolution_clock::now();
//     preprocess_time = end - start;
//     std::cout << "Postprocessing time: " << preprocess_time.count() << " ms" << std::endl;
// #endif
//     return result;
// }

// AngleType TextClassifier::postProcess(const std::vector<cv::Mat> &outputs) {
//     auto output = outputs[0];
//     cv::Mat probs;
//     cv::exp(output, probs);
//     cv::Scalar sum_exp = cv::sum(probs);
//     probs /= sum_exp[0]; // 归一化为概率分布

//     float angle_0_prob = probs.at<float>(0);
//     float angle_180_prob = probs.at<float>(1);
//     return (angle_0_prob >= angle_180_prob) ? AngleType::ANGLE_0 : AngleType::ANGLE_180;
// }