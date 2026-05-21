// my_shitty_lru.hpp
//
// Bounded-by-bytes LRU cache. Single-purpose: hardcoded to
// (std::string key  →  clip_image_f32_batch value), the result of
// clip_image_preprocess. Header-only. Internally synchronized.
//
// Sized in decimal megabytes (1 MB = 1,000,000 bytes). Byte accounting
// only counts the float buffers in the cached batches — not vector
// overhead, unique_ptr overhead, or map/list node allocations. Real RSS
// will run a few percent higher than the reported budget. Fine for
// "shitty" purposes; if you need exact RSS bounds, this isn't your cache.
//
// Thread-safety: single mutex around all ops. Compute callbacks run
// WITHOUT the lock held, so concurrent misses on different keys
// parallelize. Concurrent misses on the SAME key both run compute();
// one wins the race, the other's value is dropped. Acceptable: the
// hash key is deterministic from input, so both computes produce the
// same answer.
//
// Eviction: plain LRU (front = MRU, back = LRU). On insert, evict from
// the back until the new entry fits. A single entry larger than the
// total budget is still allowed in (the next insert will then push it
// out); this avoids deadlocking on a too-small budget.

#pragma once

#include "clip-impl.h"  // clip_image_f32_batch + its .clone()

#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

class MyShittySinglePurposeLru {
public:
    explicit MyShittySinglePurposeLru(size_t max_size_megabytes)
        : max_bytes_(max_size_megabytes * 1000ull * 1000ull)
        , cur_bytes_(0) {}

    // Snapshot accessors. Only count user-supplied byte budget (no overhead).
    size_t size_bytes() const {
        std::lock_guard<std::mutex> lk(mu_);
        return cur_bytes_;
    }
    size_t size_entries() const {
        std::lock_guard<std::mutex> lk(mu_);
        return entries_.size();
    }

    // Hit:  return clone of cached value.
    // Miss: invoke compute() (without the cache lock held), move-store the
    //       returned batch, evict LRU entries until under the byte budget,
    //       return clone.
    clip_image_f32_batch get_or_compute(
            const std::string&                       key,
            std::function<clip_image_f32_batch()>    compute) {
        // ---- Fast path: hit. ----
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = index_.find(key);
            if (it != index_.end()) {
                entries_.splice(entries_.begin(), entries_, it->second);
                return it->second->value.clone();
            }
        }

        // ---- Slow path: compute outside the lock. ----
        clip_image_f32_batch fresh = compute();
        const size_t fresh_bytes = bytes_of(fresh);

        // ---- Insert (or detect a racing insert and use that instead). ----
        std::lock_guard<std::mutex> lk(mu_);

        auto it = index_.find(key);
        if (it != index_.end()) {
            // Someone else inserted while we were computing. Drop ours.
            entries_.splice(entries_.begin(), entries_, it->second);
            return it->second->value.clone();
        }

        // Evict from LRU end until our entry fits. If our entry alone is
        // bigger than the entire budget, the loop stops at entries_.empty()
        // and we still insert it (oversize singletons are permitted).
        while (!entries_.empty() && cur_bytes_ + fresh_bytes > max_bytes_) {
            evict_lru_locked();
        }

        entries_.push_front(Entry{key, std::move(fresh), fresh_bytes});
        index_[key] = entries_.begin();
        cur_bytes_ += fresh_bytes;
        return entries_.front().value.clone();
    }

private:
    struct Entry {
        std::string          key;
        clip_image_f32_batch value;
        size_t               bytes;
    };

    using ListIt = std::list<Entry>::iterator;

    // Sum of float buffer sizes across all entries. Doesn't count struct,
    // vector, or unique_ptr overhead — see top-of-file note.
    static size_t bytes_of(const clip_image_f32_batch& b) {
        size_t n = 0;
        for (const auto& e : b.entries) {
            n += e->buf.size() * sizeof(float);
        }
        return n;
    }

    // Caller must hold mu_.
    void evict_lru_locked() {
        auto& back = entries_.back();
        cur_bytes_ -= back.bytes;
        index_.erase(back.key);
        entries_.pop_back();
    }

    mutable std::mutex                       mu_;
    std::list<Entry>                         entries_;   // front = MRU, back = LRU
    std::unordered_map<std::string, ListIt>  index_;
    const size_t                             max_bytes_;
    size_t                                   cur_bytes_;
};
