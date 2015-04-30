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

#include <vector>
#include <string>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

class cl_matrix_multiply {
private:
    cl::Buffer cl_buffer_mat1_;
    cl::Buffer cl_buffer_mat2_;
    cl::Buffer cl_buffer_result_;
    unsigned int platform_id_;
    unsigned int device_id_;
    cl::Program program_;
    cl::Kernel kernel_;
    std::vector<cl::Device> devices_;
    cl::Context context_;
    cl::CommandQueue queue_;
    cl::Event event_;
    cl_uint pitch_;
    cl_uint mat1_size_;
    cl_uint mat2_size_;
    cl_uint result_size_;
    cl_int err_;
public:
    cl_matrix_multiply(unsigned int platform, unsigned int device);
    virtual ~cl_matrix_multiply();
public:
    void init(const std::string& cl_file);
    void prepare(
        const std::vector<float>& mat1,
        const std::vector<float>& mat2,
        unsigned int pitch);
    std::chrono::duration<double> run(std::vector<float>& out);
};
