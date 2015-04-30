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

#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "cl_matrix_multiply.hpp"

using namespace boost::program_options;

int main(int ac, char** av) {
    unsigned int platform_id = 0;
    unsigned int device_id = 0;
    try {
        options_description desc("Allowed options");
        desc.add_options
        ()
        ("help,h", "produce help message")
        ("platform,p", value<unsigned int>(), "platform selection")
        ("device,d", value<unsigned int>(), "device selection");
        variable_map vm;
        store(command_line_parser(ac, av).options(desc).run(), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 1;
        }
        if (vm.count("platform")) {
            platform_id = vm["platform"].as<unsigned int>();
        }
        std::cout << "platform id     : " << platform_id << std::endl;
        if (vm.count("device")) {
            device_id = vm["device"].as<unsigned int>();
        }
        std::cout << "device id       : " << device_id << std::endl;
        // TODO instanciate and create the matrix multiply object 
    } catch (std::exception& ex) {
        std::cerr << "exception (std) : " << ex.what() << std::endl;
        return -1;
    }
    return 0;
}
