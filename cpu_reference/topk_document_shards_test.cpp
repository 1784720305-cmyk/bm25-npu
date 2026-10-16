// Independent numerical full-sort oracle for the CPU partitioned selector. No NPU calls.
#include "topk_document_shards.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <omp.h>

float FromBits(uint32_t raw)
{
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}

void Check(const std::vector<float> &input, const std::vector<float> &scores,
           const std::vector<int32_t> &ids, int64_t q, int64_t docs, int64_t k)
{
    std::vector<int32_t> all(static_cast<size_t>(docs));
    for (int64_t row = 0; row < q; ++row) {
        std::iota(all.begin(), all.end(), 0);
        const int64_t base = row*docs;
        std::sort(all.begin(), all.end(), [&](int32_t a, int32_t b) {
            return input[base+a] > input[base+b] || (input[base+a] == input[base+b] && a < b);
        });
        for (int64_t rank = 0; rank < k; ++rank) {
            const int64_t pos = row*k+rank;
            if (ids[pos] != all[rank] || std::memcmp(&scores[pos], &input[base+all[rank]], sizeof(float)) != 0)
                throw std::runtime_error("Full-sort ID or original FP32 bits mismatch");
        }
    }
}

int main()
{
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        omp_set_dynamic(0);
        omp_set_num_threads(8);
        size_t cases = 0;
        for (int64_t docs : {1, 2, 6, 31, 32, 33, 127, 128, 129, 513, 1025, 4097, 65536}) {
            for (int64_t q : {1, 3}) {
                if (docs == 65536 && q != 1) continue;
                for (int pattern = 0; pattern < 9; ++pattern) {
                    if (docs == 65536 && pattern != 0 && pattern != 4 && pattern != 8) continue;
                    std::vector<float> input(static_cast<size_t>(q*docs));
                    for (int64_t row = 0; row < q; ++row) {
                        for (int64_t doc = 0; doc < docs; ++doc) {
                            const int which = (pattern+static_cast<int>(row))%9;
                            float value = 0;
                            if (which == 0) value = 0;
                            else if (which == 1) value = 1;
                            else if (which == 2) value = doc%2 ? -0.0F : 0.0F;
                            else if (which == 3) value = static_cast<float>(doc);
                            else if (which == 4) value = static_cast<float>((doc*97+row*13)%37);
                            else if (which == 5) value = doc%3 ? 0 : FromBits(1);
                            else if (which == 6) value = doc%7 ? 0 : std::numeric_limits<float>::max();
                            else if (which == 7) value = FromBits(doc%2 ? 0x3f800001U : 0x3f800000U);
                            else value = doc == docs-1 || doc == 0 ? 99.0F : -0.0F;
                            input[row*docs+doc] = value;
                        }
                    }
                    std::vector<int64_t> ks{1, std::min<int64_t>(10, docs), std::min<int64_t>(32, docs)};
                    std::sort(ks.begin(), ks.end());
                    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
                    for (const int64_t k : ks) {
                        for (int64_t shards : {1, 2, 8}) {
                            if (shards > docs) continue;
                            std::vector<Candidate> partials(static_cast<size_t>(q*shards*k));
                            std::vector<float> output(static_cast<size_t>(q*k));
                            std::vector<int32_t> ids(static_cast<size_t>(q*k));
                            for (int repeat = 0; repeat < 3; ++repeat) {
                                // Change ID positions on the second call and return on the third.
                                if (repeat != 0) for (int64_t row = 0; row < q; ++row)
                                    std::reverse(input.begin()+row*docs, input.begin()+(row+1)*docs);
                                std::fill(output.begin(), output.end(), std::numeric_limits<float>::quiet_NaN());
                                std::fill(ids.begin(), ids.end(), -1);
                                std::fill(partials.begin(), partials.end(), Candidate{FromBits(0x7fc00000U), -1});
                                TopKDocumentShards(input, q, docs, k, shards, partials, output, ids);
                                Check(input, output, ids, q, docs, k);
                            }
                            ++cases;
                        }
                    }
                }
            }
        }
        std::cout << "CPU document-sharded Top-K passed " << cases
                  << " cases, three overwritten/alternating-ID-position calls each; full IDs/original FP32 bits match. No NPU or performance claim.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "CPU Top-K test failed: " << error.what() << '\n';
        return 1;
    }
}
