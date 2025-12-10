#include "TRTInfer.h"
// for dim
std::ostream &operator<<(std::ostream &cout, const nvinfer1::Dims &dim)
{
    for (int i = 0; i < dim.nbDims; i++)
    {
        if (i < dim.nbDims - 1)
        {
            cout << dim.d[i] << " X ";
        }
        else
            cout << dim.d[i];
    }
    return cout;
}

std::ostream &operator<<(std::ostream &cout, const nvinfer1::DataType &type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBF16:
        cout << "kBF16";
        break;
    case nvinfer1::DataType::kBOOL:
        cout << "kBOOL";
        break;
    case nvinfer1::DataType::kFLOAT:
        cout << "kFLOAT";
        break;
    case nvinfer1::DataType::kFP8:
        cout << "kFP8";
        break;
    case nvinfer1::DataType::kHALF:
        cout << "kHALF";
        break;
    case nvinfer1::DataType::kINT32:
        cout << "kINT32";
        break;
    case nvinfer1::DataType::kINT4:
        cout << "kINT4";
        break;
    case nvinfer1::DataType::kINT64:
        cout << "kINT64";
        break;
    case nvinfer1::DataType::kINT8:
        cout << "kINT8";
        break;
    case nvinfer1::DataType::kUINT8:
        cout << "kUINT8";
        break;
    default:
        break;
    }
    return cout;
}

// logger
void Logger::log(Severity severity, const char *msg) noexcept
{
    if (severity <= Severity::kERROR)
    {
        std::cout << msg << std::endl;
    }
}

// TRTInfer
TRTInfer::TRTInfer(const std::string &engine_path) : logger()
{

    load_engine(engine_path);

    context.reset(engine->createExecutionContext());

    get_InputNames();

    get_OutputNames();

    get_bindings();

    cudaStreamCreate(&stream);
}
TRTInfer::~TRTInfer()
{
    // destory stream
    cudaStreamDestroy(stream);
    // release cuda data
    for (auto &data : inputBindings)
        cudaFree(data.second);
    for (auto &data : outputBindings)
        cudaFree(data.second);
}

std::unordered_map<std::string, void *> TRTInfer::operator()(const std::unordered_map<std::string, void *> &input_blob)
{
    return infer(input_blob);
}

std::unordered_map<std::string, cv::Mat> TRTInfer::operator()(const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    return infer(input_blob);
}
void TRTInfer::load_engine(const std::string &engine_path)
{
    // read engine weights
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "Error reading engine file" << std::endl;
        throw std::runtime_error("Error reading engine file");
    }
    file.seekg(0, file.end);
    const size_t fsize = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    file.close();

    // runtime
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime)
    {
        std::cerr << "Failed to create runtime" << std::endl;
        throw std::runtime_error("Failed to create runtime");
    }

    // init plugins
    initLibNvInferPlugins(&logger, "");
    // engine
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize));
    if (!engine)
    {
        std::cerr << "Failed to create engine" << std::endl;
        throw std::runtime_error("Failed to create engine");
    }
}

void TRTInfer::get_InputNames()
{
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            // std::cout << "input tensor name : " << name
            //           << ",tensor shape : " << engine->getTensorShape(name)
            //           << ",tensor type : " << engine->getTensorDataType(name)
            //           << ",tensor format : " << engine->getTensorFormatDesc(name)
            //           << std::endl;
            input_names.emplace_back(std::string(name));
            input_size[std::string(name)] = utility::getTensorbytes(engine->getTensorShape(name), engine->getTensorDataType(name));
        }
    }
}

void TRTInfer::get_OutputNames()
{
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            // first
            // std::cout << "output tensor name : " << name
            //           << ",tensor shape : " << engine->getTensorShape(name)
            //           << ",tensor type : " << engine->getTensorDataType(name)
            //           << ",tensor format : " << engine->getTensorFormatDesc(name)
            //           << std::endl;
            // second
            output_names.emplace_back(std::string(name));
            output_size[std::string(name)] = utility::getTensorbytes(engine->getTensorShape(name), engine->getTensorDataType(name));
            // third
            // 讲tensorrt的dim类似转换为opencv的dims类型
            nvinfer1::Dims dims = engine->getTensorShape(name);
            std::vector<int> dim;
            dim.reserve(dims.nbDims);
            for(int di = 0; di < dims.nbDims; ++di)
                dim.emplace_back(static_cast<int>(dims.d[di]));
            // 填充类型
            output_shape[std::string(name)] = dim;
        }
    }
}

void TRTInfer::get_bindings()
{
    // allocate input memeory
    for (size_t i = 0; i < input_names.size(); i++)
    {
        void* ptr = utility::safeCudaMalloc(input_size[input_names[i]]);
        if(ptr == nullptr){
            std::cerr << "Failed to allocate GPU memory for input tensor: " << input_names[i]
                      << ", bytes: " << input_size[input_names[i]] << std::endl;
            throw std::runtime_error("cudaMalloc failed for input binding");
        }
        inputBindings[input_names[i]] = ptr;
    }
    // allocate output memeory
    for (size_t i = 0; i < output_names.size(); i++)
    {
        void* ptr = utility::safeCudaMalloc(output_size[output_names[i]]);
        if(ptr == nullptr){
            std::cerr << "Failed to allocate GPU memory for output tensor: " << output_names[i]
                      << ", bytes: " << output_size[output_names[i]] << std::endl;
            throw std::runtime_error("cudaMalloc failed for output binding");
        }
        outputBindings[output_names[i]] = ptr;
    }
}

std::unordered_map<std::string, void *> TRTInfer::infer(const std::unordered_map<std::string, void *> &input_blob)
{
    // input copy
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        void *cpu_ptr = input_data.second;
        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;
            size_t data_size = input_size[key];

            cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr, data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            // set the input tensor
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }

    // output set
    for (size_t i = 0; i < output_names.size(); i++)
    {
        auto it = outputBindings.find(output_names[i]);
        if(it == outputBindings.end() || it->second == nullptr){
            std::cerr << "Output binding not allocated: " << output_names[i] << std::endl;
            throw std::runtime_error("Output binding is null");
        }
        context->setOutputTensorAddress(output_names[i].c_str(), it->second);
    }

    // async execute
    if(!context->enqueueV3(stream)){
        std::cerr << "enqueueV3 failed" << std::endl;
        throw std::runtime_error("enqueueV3 failed");
    }

    // copy the gpu data to cpu data
    std::unordered_map<std::string, void *> output_blob;
    for (const auto &names : output_names)
    {
        size_t datasize = output_size[names];
        void *value_ptr = (void *)new char[datasize];
        output_blob[names] = value_ptr;
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            cudaError_t err = cudaMemcpyAsync(value_ptr, iter->second, datasize, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // waiting for the stream
    {
        cudaError_t err = cudaStreamSynchronize(stream);
        if(err != cudaSuccess){
            std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    return output_blob;
}

std::unordered_map<std::string, cv::Mat> TRTInfer::infer(const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    // input copy
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        cv::Mat cpu_ptr = input_data.second;

        // 类型转换
        if (utility::typeCv2Rt(cpu_ptr.type()) != engine->getTensorDataType(key.c_str()))
            cpu_ptr.convertTo(cpu_ptr, utility::typeRt2Cv(engine->getTensorDataType(key.c_str())));
        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;
            size_t data_size = input_size[key];

            cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr.data, data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            // set the input tensor
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }
    // output set
    for (size_t i = 0; i < output_names.size(); i++)
    {
        auto it = outputBindings.find(output_names[i]);
        if(it == outputBindings.end() || it->second == nullptr){
            std::cerr << "Output binding not allocated: " << output_names[i] << std::endl;
            throw std::runtime_error("Output binding is null");
        }
        context->setOutputTensorAddress(output_names[i].c_str(), it->second);
    }

    // async execute
    if(!context->enqueueV3(stream)){
        std::cerr << "enqueueV3 failed" << std::endl;
        throw std::runtime_error("enqueueV3 failed");
    }

    // copy the gpu data to cpu data
    std::unordered_map<std::string, cv::Mat> output_blob;
    for (const auto &names : output_names)
    {
        size_t datasize = output_size[names];
        // 创建输出数据
        cv::Mat output(
            static_cast<int>(output_shape[names].size()),
            output_shape[names].data(),
            utility::typeRt2Cv(engine->getTensorDataType(names.c_str()))
            );
        output_blob[names] = output;
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            cudaError_t err = cudaMemcpyAsync(output.data, iter->second, datasize, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // waiting for the stream
    {
        cudaError_t err = cudaStreamSynchronize(stream);
        if(err != cudaSuccess){
            std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }
    return output_blob;
}
