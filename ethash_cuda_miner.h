#pragma once

#include <time.h>
#include <libethash/ethash.h>

class ethash_cuda_miner
{
public:
	struct search_hook
	{
		// reports progress, return true to abort
		virtual bool found(uint64_t const* nonces, uint32_t count) = 0;
		virtual bool searched(uint64_t start_nonce, uint32_t count) = 0;
	};

public:
	ethash_cuda_miner();

	bool init(ethash_params const& params, ethash_h256_t const *seed, unsigned workgroup_size = 64);

	void finish();
	void hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count);
	void search(uint8_t const* header, uint64_t target, search_hook& hook);

private:
	static unsigned const c_max_search_results = 63;
	static unsigned const c_num_buffers = 2;
	static unsigned const c_hash_batch_size = 512;
	static unsigned const c_search_batch_size = 512*256;

	ethash_params m_params;
	char* m_dag;
	char* m_header;
	char* m_hash_buf[c_num_buffers];
	char* m_search_buf[c_num_buffers];
	unsigned m_workgroup_size;
};
