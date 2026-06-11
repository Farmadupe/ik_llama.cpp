// mtmd-media-memo.h
//
// Identity + token-shape memo for media containers, keyed by an FNV-1a hash
// of the encoded container bytes (image file or IMAGE_TEMPORAL container).
//
// Written by the simple image path in mtmd_tokenize after the first full
// decode + preprocess of a container; read at request-parse time
// (mtmd_helper_bitmap_init_lazy) to skip pixel decoding entirely on repeat
// sightings. Entries are tiny and never evicted; the memo dies with its
// owning mtmd_context, so it cannot survive a model swap.
//
// Writes are blind (last-write-wins): keys are content hashes, so concurrent
// writers for the same key always carry identical values.

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

struct mtmd_media_memo_entry {
    std::string id;        // pixel-content hash (same decimal-fnv format the server uses for KV tracking)
    uint32_t nx = 0;       // raw pixel dims of the decoded bitmap
    uint32_t ny = 0;
    uint32_t nz = 1;
    uint32_t n_tokens = 0; // embedding positions for the whole chunk (post temporal pooling)
    uint32_t nx_tok = 0;   // M-RoPE token grid; 0 when the model doesn't use M-RoPE
    uint32_t ny_tok = 0;
};

struct mtmd_media_memo {
    static uint64_t fnv1a(const unsigned char * data, size_t len) {
        uint64_t hash = 0xcbf29ce484222325ULL;
        for (size_t i = 0; i < len; ++i) {
            hash ^= data[i];
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }

    bool find(uint64_t key, mtmd_media_memo_entry & out) const {
        std::lock_guard<std::mutex> lk(mu);
        auto it = map.find(key);
        if (it == map.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

    void put(uint64_t key, mtmd_media_memo_entry entry) {
        std::lock_guard<std::mutex> lk(mu);
        map[key] = std::move(entry);
    }

    size_t count() const {
        std::lock_guard<std::mutex> lk(mu);
        return map.size();
    }

    mutable std::mutex mu;
    std::unordered_map<uint64_t, mtmd_media_memo_entry> map;
};

// implemented in mtmd.cpp; the memo is owned by the mtmd_context
struct mtmd_context;
mtmd_media_memo * mtmd_ctx_media_memo(struct mtmd_context * ctx);
