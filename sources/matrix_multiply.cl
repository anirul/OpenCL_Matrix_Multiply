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

kernel void matrix_multiply(
    global const float* mat1,
    global const float* mat2,
    global float* result,
    uint pitch)
{
    const int pos_out = get_global_id(1) * get_global_size(0) + get_global_id(0);
    const int pos_in1 = get_global_id(0) * pitch;
    const int pos_in2 = get_global_id(1) * pitch;
    result[pos_out] = 0;
    for (uint i = 0; i < pitch; ++i) {
        result[pos_out] += mat1[pos_in1 + i] * mat2[pos_in2 + i];
    }
}

#define BLOCK_SIZE 16

kernel void matrix_multiply_block(
    global float* mat1,
    global float* mat2,
    global float* result,
    uint pitch)
{
    // workgroup id
    const int i = get_group_id(0);
    const int j = get_group_id(1);
    // workitem id
    const int idX = get_local_id(0);
    const int idY = get_local_id(1);
    //matrices dimensions
    int p = get_global_size(0);
    int r = get_global_size(1);
    // number of submatrices to be processes by each worker (Q dimension)
    const int numSubMat = pitch / BLOCK_SIZE;
    float4 resp = (float4)(0, 0, 0, 0);
    local float A[BLOCK_SIZE][BLOCK_SIZE];
    local float B[BLOCK_SIZE][BLOCK_SIZE];
    for (int k = 0; k < numSubMat; k++) {
        // copy submatrices to local memory. each worker copies one element
        // Notice that A[i,k] accesses element starting from M[BLOCK_SIZE * i, BLOCK_SIZE * j]
        A[idX][idY] = mat1[BLOCK_SIZE*i + idX + p*(BLOCK_SIZE*k+idY)];
        B[idX][idY] = mat2[BLOCK_SIZE*k + idX + pitch*(BLOCK_SIZE*j+idY)];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k2 = 0; k2 < BLOCK_SIZE; k2+=4)
        {
            float4 temp1=(float4)(A[idX][k2],A[idX][k2+1],A[idX][k2+2],A[idX][k2+3]);
            float4 temp2=(float4)(B[k2][idY],B[k2+1][idY],B[k2+2][idY],B[k2+3][idY]);
            resp += temp1 * temp2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    result[BLOCK_SIZE*i + idX + p*(BLOCK_SIZE*j+idY)] = resp.x+resp.y+resp.z+resp.w;
}
