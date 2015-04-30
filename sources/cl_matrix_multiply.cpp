/*
 * Copyright (c) 2015, Frederic Dubouchet
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Calodox nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Frederic Dubouchet ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Frederic DUBOUCHET BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "cl_matrix_multiply.hpp"

cl_matrix_multiply::cl_matrix_multiply(
    unsigned int platform,
    unsigned int device) :
platform_id_(platform),
device_id_(device)
{
    std::vector<cl::Platform> platforms;
    auto err = cl::Platform::get(&platforms);
    if (platforms.size() <= platform_id_)
        throw std::runtime_error("unknown platform");
    err = platforms[platform_id_].getDevices(CL_DEVICE_TYPE_ALL, &devices_);
    int i = 0;
    for (auto device : devices_) {
        std::cout
            << "device name [" << i << "] : "
            << device.getInfo<CL_DEVICE_NAME>()
            << std::endl;
        i++;
    }
    if (devices_.size() <= device_id_)
        throw std::runtime_error("unknown device");
    std::cout << "device used     : " << device_id_ << std::endl;
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(platforms[platform_id_])(),
        0
    };
    context_ = cl::Context(CL_DEVICE_TYPE_ALL, properties);
    devices_ = context_.getInfo<CL_CONTEXT_DEVICES>();
    queue_ = cl::CommandQueue(context_, devices_[device_id_], 0, &err);
}

cl_matrix_multiply::~cl_matrix_multiply() {}

void cl_matrix_multiply::init(const std::string& cl_file) {
    std::ifstream ifs(cl_file);
    if (!ifs.is_open())
        throw std::runtime_error("could not open file : " + cl_file);
    std::string kernel_source(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());
    cl::Program::Sources sources(
        1,
        std::make_pair(
            kernel_source.c_str(),
            kernel_source.size()));
    program_ = cl::Program(context_, sources);
    try {
        auto err = program_.build(devices_);
    } catch (cl::Error er) {
        std::cerr
            << "build status    : "
            << program_.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(
                devices_[device_id_])
            << std::endl;
        std::cerr
            << "build options   : "
            << program_.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(
                devices_[device_id_])
            << std::endl;
        std::cerr
            << "build log       : "
            << program_.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                devices_[device_id_])
            << std::endl;
        throw er;
    }
}

void cl_matrix_multiply::prepare(
    const std::vector<float>& mat1,
    const std::vector<float>& mat2,
    unsigned int pitch)
{
    pitch_ = pitch;
    if (mat1.size() % pitch != 0)
        throw std::runtime_error("matrix 1 should be dividable by pitch");
    if (mat2.size() % pitch != 0)
        throw std::runtime_error("matrix 2 should be dividable by pitch");
    mat1_size_ = mat1.size();
    mat2_size_ = mat2.size();
    result_size_ = (mat1_size_ / pitch_) * (mat2_size_ / pitch_);
    cl_buffer_mat1_ = cl::Buffer(
        context_,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * mat1.size(),
        (void*)&mat1[0]);
    cl_buffer_mat2_ = cl::Buffer(
        context_,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_float) * mat2.size(),
        (void*)&mat2[0]);
    cl_buffer_result_ = cl::Buffer(
        context_,
        CL_MEM_READ_WRITE,
        sizeof(cl_float) * result_size_);
    kernel_ = cl::Kernel(
        program_,
        "matrix_multiply_block",
        &err_);
}

std::chrono::duration<double> cl_matrix_multiply::run(std::vector<float>& out) {
    kernel_.setArg(0, cl_buffer_mat1_);
    kernel_.setArg(1, cl_buffer_mat2_);
    kernel_.setArg(2, cl_buffer_result_);
    kernel_.setArg(3, pitch_);
    queue_.finish();
    auto start = std::chrono::system_clock::now();
    err_ = queue_.enqueueNDRangeKernel(
        kernel_,
        cl::NullRange,
        cl::NDRange(mat1_size_ / pitch_, mat2_size_ / pitch_),
        cl::NDRange(16, 16),
        nullptr,
        &event_);
    queue_.finish();
    auto end = std::chrono::system_clock::now();
    //TODO copy the result in out
    return end - start;
}
