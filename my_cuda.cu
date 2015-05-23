/*
  This file is part of c-ethash.

  c-ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  c-ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** 
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015
*
* CUDA port by Grigore Lupescu <grigore.lupescu@gmail.com>
*/

#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <queue>
#include <vector>
#include <libethash/util.h>
#include <string>
#include "ethash_cuda_miner.h"

#include <vector_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#define ETHASH_BYTES 32

#undef min
#undef max

#define DAG_SIZE 262688
#define MAX_OUTPUTS 63
#define GROUP_SIZE 64
#define ACCESSES 64

/* APPLICATION SETTINGS */
#define __ENDIAN_LITTLE__	1

#define THREADS_PER_HASH (8)
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)

#define FNV_PRIME	0x01000193

__constant__ uint2 Keccak_f1600_RC[24] = {
	{0x00000001, 0x00000000},
	{0x00008082, 0x00000000},
	{0x0000808a, 0x80000000},
	{0x80008000, 0x80000000},
	{0x0000808b, 0x00000000},
	{0x80000001, 0x00000000},
	{0x80008081, 0x80000000},
	{0x00008009, 0x80000000},
	{0x0000008a, 0x00000000},
	{0x00000088, 0x00000000},
	{0x80008009, 0x00000000},
	{0x8000000a, 0x00000000},
	{0x8000808b, 0x00000000},
	{0x0000008b, 0x80000000},
	{0x00008089, 0x80000000},
	{0x00008003, 0x80000000},
	{0x00008002, 0x80000000},
	{0x00000080, 0x80000000},
	{0x0000800a, 0x00000000},
	{0x8000000a, 0x80000000},
	{0x80008081, 0x80000000},
	{0x00008080, 0x80000000},
	{0x80000001, 0x00000000},
	{0x80008008, 0x80000000},
};

/*----------------------------------------------------------------------
*	HOST/TARGET FUNCTIONS
*---------------------------------------------------------------------*/

/******************************************
* FUNCTION: keccak_f1600_round
*******************************************/

__device__ __host__ uint2 operator ^ (const uint2 &f, const uint2 &s) {
       uint2 result;
       result.x = f.x ^ s.x;
       result.y = f.y ^ s.y;

       return result;
}
    
__device__ __host__ uint2 bitselect(uint2 a, uint2 b, uint2 cmp) {
    uint2 result;
    result.x = (a.x & (~cmp.x)) | (b.x & cmp.x);
    result.y = (a.y & (~cmp.y)) | (b.y & cmp.y);

    return result;
}

__device__ __host__ void keccak_f1600_round(uint2* a, uint r, uint out_size)
{
	#if !__ENDIAN_LITTLE__
		for (uint i = 0; i != 25; ++i)
			a[i] = make_uint2(a[i].y, a[i].x);
	#endif
    
   /*** DONE - ASCHW4 **************************
	* Implementare Keccak/SHA3 pornind de la 
	* resurse SHA3, cod OpenCL/C/Python/Go
	*********************************************/
   
    uint2 b[25];
    uint2 t;
    
    // Theta
	b[0] = a[0] ^ a[5] ^ a[10] ^ a[15] ^ a[20];
	b[1] = a[1] ^ a[6] ^ a[11] ^ a[16] ^ a[21];
	b[2] = a[2] ^ a[7] ^ a[12] ^ a[17] ^ a[22];
	b[3] = a[3] ^ a[8] ^ a[13] ^ a[18] ^ a[23];
	b[4] = a[4] ^ a[9] ^ a[14] ^ a[19] ^ a[24];
	t = b[4] ^ make_uint2(b[1].x << 1 | b[1].y >> 31, b[1].y << 1 | b[1].x >> 31);
	a[0] = a[0] ^ t;
	a[5] = a[5] ^ t;
	a[10] = a[10] ^ t;
	a[15] = a[15] ^ t;
	a[20] = a[20] ^ t;
	t = b[0] ^ make_uint2(b[2].x << 1 | b[2].y >> 31, b[2].y << 1 | b[2].x >> 31);
	a[1] = a[1] ^ t;
	a[6] = a[6] ^ t;
	a[11] = a[11] ^ t;
	a[16] = a[16] ^ t;
	a[21] = a[20] ^ t;
	t = b[1] ^ make_uint2(b[3].x << 1 | b[3].y >> 31, b[3].y << 1 | b[3].x >> 31);
	a[2] = a[2] ^ t;
	a[7] = a[7] ^ t;
	a[12] = a[12] ^ t;
	a[17] = a[17] ^ t;
	a[22] = a[22] ^ t;
	t = b[2] ^ make_uint2(b[4].x << 1 | b[4].y >> 31, b[4].y << 1 | b[4].x >> 31);
	a[3] = a[3] ^ t;
	a[8] = a[8] ^ t;
	a[13] = a[13] ^ t;
	a[18] = a[18] ^ t;
	a[23] = a[23] ^ t;
	t = b[3] ^ make_uint2(b[0].x << 1 | b[0].y >> 31, b[0].y << 1 | b[0].x >> 31);
	a[4] = a[4] ^ t;
	a[9] = a[9] ^ t;
	a[14] = a[14] ^ t;
	a[19] = a[19] ^ t;
	a[24] = a[24] ^ t;

	// Rho Pi
	b[0] = a[0];
	b[10] = make_uint2(a[1].x << 1 | a[1].y >> 31, a[1].y << 1 | a[1].x >> 31);
	b[7] = make_uint2(a[10].x << 3 | a[10].y >> 29, a[10].y << 3 | a[10].x >> 29);
	b[11] = make_uint2(a[7].x << 6 | a[7].y >> 26, a[7].y << 6 | a[7].x >> 26);
	b[17] = make_uint2(a[11].x << 10 | a[11].y >> 22, a[11].y << 10 | a[11].x >> 22);
	b[18] = make_uint2(a[17].x << 15 | a[17].y >> 17, a[17].y << 15 | a[17].x >> 17);
	b[3] = make_uint2(a[18].x << 21 | a[18].y >> 11, a[18].y << 21 | a[18].x >> 11);
	b[5] = make_uint2(a[3].x << 28 | a[3].y >> 4, a[3].y << 28 | a[3].x >> 4);
	b[16] = make_uint2(a[5].y << 4 | a[5].x >> 28, a[5].x << 4 | a[5].y >> 28);
	b[8] = make_uint2(a[16].y << 13 | a[16].x >> 19, a[16].x << 13 | a[16].y >> 19);
	b[21] = make_uint2(a[8].y << 23 | a[8].x >> 9, a[8].x << 23 | a[8].y >> 9);
	b[24] = make_uint2(a[21].x << 2 | a[21].y >> 30, a[21].y << 2 | a[21].x >> 30);
	b[4] = make_uint2(a[24].x << 14 | a[24].y >> 18, a[24].y << 14 | a[24].x >> 18);
	b[15] = make_uint2(a[4].x << 27 | a[4].y >> 5, a[4].y << 27 | a[4].x >> 5);
	b[23] = make_uint2(a[15].y << 9 | a[15].x >> 23, a[15].x << 9 | a[15].y >> 23);
	b[19] = make_uint2(a[23].y << 24 | a[23].x >> 8, a[23].x << 24 | a[23].y >> 8);
	b[13] = make_uint2(a[19].x << 8 | a[19].y >> 24, a[19].y << 8 | a[19].x >> 24);
	b[12] = make_uint2(a[13].x << 25 | a[13].y >> 7, a[13].y << 25 | a[13].x >> 7);
	b[2] = make_uint2(a[12].y << 11 | a[12].x >> 21, a[12].x << 11 | a[12].y >> 21);
	b[20] = make_uint2(a[2].y << 30 | a[2].x >> 2, a[2].x << 30 | a[2].y >> 2);
	b[14] = make_uint2(a[20].x << 18 | a[20].y >> 14, a[20].y << 18 | a[20].x >> 14);
	b[22] = make_uint2(a[14].y << 7 | a[14].x >> 25, a[14].x << 7 | a[14].y >> 25);
	b[9] = make_uint2(a[22].y << 29 | a[22].x >> 3, a[22].x << 29 | a[22].y >> 3);
	b[6] = make_uint2(a[9].x << 20 | a[9].y >> 12, a[9].y << 20 | a[9].x >> 12);
	b[1] = make_uint2(a[6].y << 12 | a[6].x >> 20, a[6].x << 12 | a[6].y >> 20);

	// Chi
	a[0] = bitselect(b[0] ^ b[2], b[0], b[1]);
	a[1] = bitselect(b[1] ^ b[3], b[1], b[2]);
	a[2] = bitselect(b[2] ^ b[4], b[2], b[3]);
	a[3] = bitselect(b[3] ^ b[0], b[3], b[4]);
	if (out_size >= 4)
	{
		a[4] = bitselect(b[4] ^ b[1], b[4], b[0]);
		a[5] = bitselect(b[5] ^ b[7], b[5], b[6]);
		a[6] = bitselect(b[6] ^ b[8], b[6], b[7]);
		a[7] = bitselect(b[7] ^ b[9], b[7], b[8]);
		a[8] = bitselect(b[8] ^ b[5], b[8], b[9]);
		if (out_size >= 8)
		{
			a[9] = bitselect(b[9] ^ b[6], b[9], b[5]);
			a[10] = bitselect(b[10] ^ b[12], b[10], b[11]);
			a[11] = bitselect(b[11] ^ b[13], b[11], b[12]);
			a[12] = bitselect(b[12] ^ b[14], b[12], b[13]);
			a[13] = bitselect(b[13] ^ b[10], b[13], b[14]);
			a[14] = bitselect(b[14] ^ b[11], b[14], b[10]);
			a[15] = bitselect(b[15] ^ b[17], b[15], b[16]);
			a[16] = bitselect(b[16] ^ b[18], b[16], b[17]);
			a[17] = bitselect(b[17] ^ b[19], b[17], b[18]);
			a[18] = bitselect(b[18] ^ b[15], b[18], b[19]);
			a[19] = bitselect(b[19] ^ b[16], b[19], b[15]);
			a[20] = bitselect(b[20] ^ b[22], b[20], b[21]);
			a[21] = bitselect(b[21] ^ b[23], b[21], b[22]);
			a[22] = bitselect(b[22] ^ b[24], b[22], b[23]);
			a[23] = bitselect(b[23] ^ b[20], b[23], b[24]);
			a[24] = bitselect(b[24] ^ b[21], b[24], b[20]);
		}
	}

	// Iota
	a[0] = a[0] ^ Keccak_f1600_RC[r];
	
	#if !__ENDIAN_LITTLE__
		for (uint i = 0; i != 25; ++i)
			a[i] = make_uint2(a[i].y, a[i].x);
	#endif
}

/******************************************
* FUNCTION: keccak_f1600_no_absorb
*******************************************/
__host__ __device__ void keccak_f1600_no_absorb(ulong* a, uint in_size, uint out_size, uint isolate)
{
	
	for (uint i = in_size; i != 25; ++i)
	{
		a[i] = 0;
	}

#if __ENDIAN_LITTLE__
	a[in_size] = a[in_size] ^ 0x0000000000000001;
	a[24-out_size*2] = a[24-out_size*2] ^ 0x8000000000000000;
#else
	a[in_size] = a[in_size] ^ 0x0100000000000000;
	a[24-out_size*2] = a[24-out_size*2] ^ 0x0000000000000080;
#endif

	// Originally I unrolled the first and last rounds to interface
	// better with surrounding code, however I haven't done this
	// without causing the AMD compiler to blow up the VGPR usage.
	uint r = 0;
	do
	{
		// This dynamic branch stops the AMD compiler unrolling the loop
		// and additionally saves about 33% of the VGPRs, enough to gain another
		// wavefront. Ideally we'd get 4 in flight, but 3 is the best I can
		// massage out of the compiler. It doesn't really seem to matter how
		// much we try and help the compiler save VGPRs because it seems to throw
		// that information away, hence the implementation of keccak here
		// doesn't bother.
		if (isolate)
		{
			/*** DONE - ASCHW4: call to keccak_f1600_round */
			 keccak_f1600_round((uint2*)a, r++, 25);
		}
	}
	while (r < 23);

	// final round optimised for digest size
	/*** DONE - ASCHW4: call to keccak_f1600_round */
	 keccak_f1600_round((uint2*)a, r++, out_size); 
}

#define copy(dst, src, count) for (uint i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }

#define countof(x) (sizeof(x) / sizeof(x[0]))

__host__ __device__ uint fnv(uint x, uint y)
{
	return x * FNV_PRIME ^ y;
}

__host__ __device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 res;

	res.x = a.x * FNV_PRIME ^ b.x;
	res.y = a.y * FNV_PRIME ^ b.y;
	res.z = a.z * FNV_PRIME ^ b.z;
	res.w = a.w * FNV_PRIME ^ b.w;

	return res;
}

__device__ __host__ uint fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

typedef union
{
	ulong ulongs[32 / sizeof(ulong)];
	uint uints[32 / sizeof(uint)];
} hash32_t;

typedef union
{
	ulong ulongs[64 / sizeof(ulong)];
	uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union
{
	uint uints[128 / sizeof(uint)];
	uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

/******************************************
* FUNCTION: init_hash
*******************************************/
__device__ __host__ hash64_t init_hash( hash32_t const* header, ulong nonce, uint isolate)
{
	hash64_t init;
	uint const init_size = countof(init.ulongs);
	uint const hash_size = countof(header->ulongs);

	// sha3_512(header .. nonce)
	ulong state[25];
	copy(state, header->ulongs, hash_size);
	state[hash_size] = nonce;
	keccak_f1600_no_absorb(state, hash_size + 1, init_size, isolate);

	copy(init.ulongs, state, init_size);
	return init;
}

/******************************************
* FUNCTION: inner_loop
*******************************************/
__device__ uint inner_loop(uint4 init, uint thread_id, uint* share, hash128_t const* g_dag, uint isolate)
{
	uint4 mix = init;

	// share init0
	if (thread_id == 0)
		*share = mix.x;

	/*** DONE - ASCHW4: uncomment when function qualifiers are OK!! */
	__syncthreads();
	uint init0 = *share;

	uint a = 0;
	do
	{
		bool update_share = thread_id == (a/4) % THREADS_PER_HASH;

		#pragma unroll
		for (uint i = 0; i != 4; ++i)
		{
			if (update_share)
			{
				uint m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a+i), m[i]) % DAG_SIZE;
			}
			/*** DONE - ASCHW4: uncomment when function qualifiers are OK!! */
			__syncthreads();

			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
		}
	}
	while ((a += 4) != (ACCESSES & isolate));

	return fnv_reduce(mix);
}

/******************************************
* FUNCTION: final_hash
*******************************************/
__host__ __device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix, uint isolate)
{
	ulong state[25];

	hash32_t hash;
	uint const hash_size = countof(hash.ulongs);
	uint const init_size = countof(init->ulongs);
	uint const mix_size = countof(mix->ulongs);

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->ulongs, init_size);
	copy(state + init_size, mix->ulongs, mix_size);
	keccak_f1600_no_absorb(state, init_size+mix_size, hash_size, isolate);

	// copy out
	copy(hash.ulongs, state, hash_size);
	return hash;
}


typedef union
{
	struct
	{
		hash64_t init;
		uint pad; // avoid lds bank conflicts
	};
	hash32_t mix;
} compute_hash_share;

/******************************************
* FUNCTION: compute_hash_simple
* INFO: no optimisations
*******************************************/
__device__ __host__ hash32_t compute_hash_simple(
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong nonce,
	uint isolate
	)
{
	hash64_t init = init_hash(g_header, nonce, isolate);

	hash128_t mix;
	for (uint i = 0; i != countof(mix.uint4s); ++i)
	{
		mix.uint4s[i] = init.uint4s[i % countof(init.uint4s)];
	}
	
	uint mix_val = mix.uints[0];
	uint init0 = mix.uints[0];
	uint a = 0;
	do
	{
		uint pi = fnv(init0 ^ a, mix_val) % DAG_SIZE;
		uint n = (a+1) % countof(mix.uints);

		for (uint i = 0; i != countof(mix.uints); ++i)
		{
			mix.uints[i] = fnv(mix.uints[i], g_dag[pi].uints[i]);
			mix_val = i == n ? mix.uints[i] : mix_val;
		}
	}
	while (++a != (ACCESSES & isolate));

	// reduce to output
	hash32_t fnv_mix;
	for (uint i = 0; i != countof(fnv_mix.uints); ++i)
	{
		fnv_mix.uints[i] = fnv_reduce(mix.uint4s[i]);
	}
	
	return final_hash(&init, &fnv_mix, isolate);
}

/******************************************
* FUNCTION: ethash_hash
*******************************************/
__global__ void ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
	)
{
	/*** DONE - ASCHW4: compute global thread id from blockIdx, blockDim and threadIdx*/

    int xindex = threadIdx.x + blockIdx.x * blockDim.x;
    int yindex = threadIdx.y + blockIdx.y * blockDim.y;
    
    uint const gid = xindex + (gridDim.x * gridDim.y * yindex);
	
	g_hashes[gid] = compute_hash_simple(g_header, g_dag, start_nonce + gid, isolate);
}

/******************************************
* FUNCTION: ethash_search
*******************************************/
__global__ void ethash_search(
	uint* g_output,
	 hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
	)
{
	/*** DONE - ASCHW4: compute global thread id from blockIdx, blockDim and threadIdx*/
	int xindex = threadIdx.x + blockIdx.x * blockDim.x;
    int yindex = threadIdx.y + blockIdx.y * blockDim.y;
    
    uint const gid = xindex + (gridDim.x * gridDim.y * yindex);

	hash32_t hash = compute_hash_simple(g_header, g_dag, start_nonce + gid, isolate);

	if (hash.ulongs[countof(hash.ulongs)-1] < target)
	{
		/*** DONE - ASCHW4: use atomicInc when function qualifiers are OK !!! */
		uint slot = min(MAX_OUTPUTS, atomicInc(&g_output[0], 1) + 1);
		//uint slot = min(MAX_OUTPUTS, (++g_output[0]) + 1);
		g_output[slot] = gid;
	}
}

/*----------------------------------------------------------------------
*	HOST ONLY FUNCTIONS
*---------------------------------------------------------------------*/
__host__ ethash_cuda_miner::ethash_cuda_miner()
{
}

__host__ void ethash_cuda_miner::finish()
{
}

/******************************************
* FUNCTION: init
*******************************************/
__host__ bool ethash_cuda_miner::init(ethash_params const& params, ethash_h256_t const *seed, unsigned workgroup_size)
{
	// store params
	m_params = params;

	// use requested workgroup size, but we require multiple of 8
	m_workgroup_size = ((workgroup_size + 7) / 8) * 8;

	// create buffer for dag
	/*** DONE - ASCHW4: Allocate using cudaMalloc for m_dag, size m_params.full_size */
    cudaMalloc((void **)&m_dag, m_params.full_size);
	// create buffer for header
	/*** DONE - ASCHW4: Allocate using cudaMalloc for m_header, size 32 */
    cudaMalloc((void **)&m_header, 32);
	// compute dag on CPU
	{
		void* cache_mem = malloc(m_params.cache_size + 63);
		ethash_cache cache;
		cache.mem = (void*)(((uintptr_t)cache_mem + 63) & ~63);
		ethash_mkcache(&cache, &m_params, seed);

		// if this throws then it's because we probably need to subdivide the dag uploads for compatibility
		char* dag_ptr = (char*) malloc(m_params.full_size);
		ethash_compute_full_data(dag_ptr, &m_params, &cache);

		/*** DONE - ASCHW4: Copy memory RAM->VRAM, SRC:dag_ptr, DST:m_dag, SIZE:m_params.full_size */
		cudaMemcpy(m_dag, dag_ptr, m_params.full_size, cudaMemcpyHostToDevice);
		delete[] dag_ptr;

		free(cache_mem);
	}

	// create mining buffers
	for (unsigned i = 0; i != c_num_buffers; ++i)
	{
		/*** DDONE - ASCHW4: Allocate memory on device/VRAM
		* m_hash_buf[i], SIZE: 32*c_hash_batch_size
		* m_search_buf[i], SIZE: (c_max_search_results + 1) * sizeof(uint32_t) */
        cudaMalloc((void **)&m_hash_buf[i], 32 * c_hash_batch_size);
        cudaMalloc((void **)&m_search_buf[i], (c_max_search_results + 1) * sizeof(uint32_t));
	}
	return true;
}

/******************************************
* FUNCTION: hash
*******************************************/
struct pending_batch
{
	unsigned base;
	unsigned count;
	unsigned buf;
};

__host__ void ethash_cuda_miner::hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count)
{
	std::queue<pending_batch> pending;

	/*** DONE - ASCHW4: Copy memory RAM->VRAM, SRC:header, DST:m_header, SIZE:32 */
    cudaMemcpy(m_header, header, 32, cudaMemcpyHostToDevice);
	unsigned buf = 0;
	for (unsigned i = 0; i < count || !pending.empty(); )
	{
		// how many this batch
		if (i < count)
		{
			unsigned const this_count = std::min(count - i, c_hash_batch_size);
			unsigned const batch_count = std::max(this_count, m_workgroup_size);

			pending_batch temp_pending_batch;
			temp_pending_batch.base = i;
			temp_pending_batch.count = this_count;
			temp_pending_batch.buf = buf;

			// execute it!
			/*** DONE - ASCHW4: call to ethash_hash */
			/* EXEC batch_count instances, 
			* ARGS to pass: m_hash_buf[buf], m_header, m_dag, nonce, ~0U */
            ethash_hash<<<batch_count / 16, 16>>>((hash32_t *)m_hash_buf[buf], (hash32_t const *)m_header, (hash128_t const *)m_dag, nonce, ~0U);
			
            pending.push(temp_pending_batch);
			i += this_count;
			buf = (buf + 1) % c_num_buffers;
		}

		// read results
		if (i == count || pending.size() == c_num_buffers)
		{
			pending_batch const& batch = pending.front();

			// could use pinned host pointer instead, but this path isn't that important.
			/*** DONE - ASCHW4: Copy memory VRAM->RAM, SRC:m_hash_buf[batch.buf], 
			* DST:ret + batch.base*ETHASH_BYTES, SIZE:batch.count*ETHASH_BYTES */
            cudaMemcpy((ret + batch.base * ETHASH_BYTES), m_hash_buf[batch.buf], (batch.count * ETHASH_BYTES), cudaMemcpyDeviceToHost);

			pending.pop();
		}
	}
}

/******************************************
* FUNCTION: search
*******************************************/
struct pending_batch_search
{
	uint64_t start_nonce;
	unsigned buf;
};

__host__ void ethash_cuda_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{
	std::queue<pending_batch_search> pending;
	static uint32_t const c_zero = 0;
	uint32_t* results = (uint32_t*)malloc((1+c_max_search_results) * sizeof(uint32_t));

	// update header constant buffer
	/*** DONE - ASCHW4: Copy memory RAM->VRAM, SRC:header, DST:m_header, SIZE:32 */
    cudaMemcpy(m_header, header, 32, cudaMemcpyHostToDevice);
	for (unsigned i = 0; i != c_num_buffers; ++i)
		cudaMemcpy( m_search_buf[i], &c_zero, 4, cudaMemcpyHostToDevice);
 
	unsigned buf = 0;
	for (uint64_t start_nonce = 0; ; start_nonce += c_search_batch_size)
	{
		/*** DONE - ASCHW4: call to ethash_search */
		/* EXEC c_search_batch_size instances, 
		* ARGS to pass: m_search_buf[buf], m_header, m_dag, start_nonce, target, ~0U */
        ethash_search<<<c_search_batch_size / 16, 16>>>((uint *)m_search_buf[buf], (hash32_t const *)m_header, (hash128_t const *)m_dag, start_nonce, target, ~0U);

		pending_batch_search temp_pending_batch;
		temp_pending_batch.start_nonce = start_nonce;
		temp_pending_batch.buf = buf;

		pending.push(temp_pending_batch);
		buf = (buf + 1) % c_num_buffers;
		
		// read results
		if (pending.size() == c_num_buffers)
		{
			pending_batch_search const& batch = pending.front();
            
			// could use pinned host pointer instead
			/*** DONE - ASCHW4: Copy memory VRAM->RAM, SRC:m_search_buf[batch.buf], 
			 * DST:results, SIZE:(1+c_max_search_results) * sizeof(uint32_t) */
			cudaMemcpy (results, m_search_buf[batch.buf], (1+c_max_search_results) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			unsigned num_found = std::min(results[0], c_max_search_results);

			uint64_t nonces[c_max_search_results];
			for (unsigned i = 0; i != num_found; ++i)
				nonces[i] = batch.start_nonce + results[i+1];

			bool exit = num_found && hook.found(nonces, num_found);
			exit |= hook.searched(batch.start_nonce, c_search_batch_size); // always report searched before exit
			if (exit)
				break;
				
			// end search prematurely due to poor performance
			if(start_nonce == 524288)
				break;

			// reset search buffer if we're still going
			if (num_found)
				cudaMemcpy(m_search_buf[batch.buf], &c_zero, 4, cudaMemcpyHostToDevice);

			pending.pop();
		}
	}
	
	delete[] results;
}

