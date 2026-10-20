#pragma once
// Timing harness only. Both backends include their unchanged, accepted Search implementation.
#include "compiled_benchmark_sources.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bm25bench {
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static_assert(sizeof(float)==4 && sizeof(int32_t)==4 && sizeof(int64_t)==8 && std::numeric_limits<float>::is_iec559,
              "Benchmark requires IEEE FP32 and fixed INT32/INT64 widths");
inline void Check(bool ok,const char *message) { if (!ok) throw std::runtime_error(message); }
inline int64_t Number(const char *text,int64_t low,int64_t high)
{
    size_t used=0; const auto value=std::stoll(text,&used);
    Check(used==std::strlen(text) && value>=low && value<=high,"Invalid benchmark numeric argument"); return value;
}
template<class T> bool Equal(const std::vector<T> &a,const std::vector<T> &b)
{
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0;
}
struct Sample { int cycle,which; double callMs,internalMs; uint64_t queryH2DBytes,topkD2HBytes; };
struct Measurement { std::vector<Sample> samples; int warmup=0,repeats=0; };

template<class Session,class Poison,class Save,class Transfers>
Measurement Measure(Session &session,const bm25search::Queries &a,const bm25search::Queries &b,
                    int warmup,int repeats,Poison poison,Save save,Transfers transfers)
{
    Check(!a.empty() && a.size()==b.size() && a!=b,"A/B must be different raw query-ID batches with equal Q");
    const std::array<const bm25search::Queries *,2> queries{&a,&b};
    std::cout << "stage=before_measurement_poisoned_full_checkpoints" << std::endl;
    poison(); auto goldA=session.Search(a); save("before_A",a,goldA);
    poison(); auto goldB=session.Search(b); save("before_B",b,goldB);
    const std::array<const std::vector<float> *,2> scores{&goldA.scores,&goldB.scores};
    const std::array<const std::vector<int32_t> *,2> ids{&goldA.ids,&goldB.ids};
    Measurement measured; measured.warmup=warmup; measured.repeats=repeats;
    measured.samples.reserve(static_cast<size_t>(2)*repeats);
    auto validate=[&](int which,const auto &result) {
        Check(Equal(result.scores,*scores[which]) && Equal(result.ids,*ids[which]),"Alternating Search Top-K bits changed");
    };
    std::cout << "stage=warmup_alternating_AB_BA" << std::endl;
    for (int cycle=0;cycle<warmup;++cycle)
        for (int slot=0;slot<2;++slot) {
            const int which=slot^(cycle&1); auto result=session.Search(*queries[which]); validate(which,result);
        }
    std::cout << "stage=measure_no_debug_io_or_poison_between_calls" << std::endl;
    for (int cycle=0;cycle<repeats;++cycle)
        for (int slot=0;slot<2;++slot) {
            const int which=slot^(cycle&1);
            const auto start=Clock::now();
            auto result=session.Search(*queries[which]);
            const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            // STOP above: CPU already owns final vectors. Validation/storage/destruction below
            // are outside the measured call, identically for CPU and NPU.
            Check(std::isfinite(elapsed) && elapsed>0 && std::isfinite(result.searchMs) &&
                  result.searchMs>0 && elapsed>=result.searchMs,"Invalid caller/internal timing");
            validate(which,result);
            const auto bytes=transfers(result);
            measured.samples.push_back(Sample{cycle,which,elapsed,result.searchMs,bytes[0],bytes[1]});
        }
    std::cout << "stage=after_measurement_poisoned_full_checkpoints" << std::endl;
    poison(); auto lastA=session.Search(a); validate(0,lastA); save("after_A",a,lastA);
    poison(); auto lastB=session.Search(b); validate(1,lastB); save("after_B",b,lastB);
    return measured;
}

inline void WriteCommon(std::ostream &out,const char *backend,const Measurement &data,
                        int64_t docs,int64_t q,int64_t k,int64_t partitions,int64_t capq,int64_t capwords,
                        uint64_t indexBytes,double coldMs)
{
    out << std::setprecision(17) << "{\n\"status\":\"search_benchmark_calls_completed_checks_pending\",\n"
        << "\"backend\":\"" << backend << "\",\"docs\":" << docs << ",\"queries\":" << q << ",\"k\":" << k
        << ",\"partitions\":" << partitions << ",\"capacity_queries\":" << capq << ",\"capacity_words\":" << capwords << ",\n"
        << "\"input_contract\":\"raw_grouped_INT32_word_IDs_in_CPU_memory\",\n"
        << "\"timer_policy\":\"caller steady_clock before Search through returned CPU Top-K\",\n"
        << "\"index_load_batches\":1,\"index_bytes\":" << indexBytes << ",\"cold_init_ms\":" << coldMs << ",\n"
        << "\"cold_synthetic_empty_queries\":" << capq << ",\"checkpoint_search_calls\":4,\n"
        << "\"warmup_cycles\":" << data.warmup << ",\"warmup_search_calls\":" << 2*data.warmup
        << ",\"warmup_topk_bit_checks\":" << 2*data.warmup << ",\n"
        << "\"measured_cycles\":" << data.repeats << ",\"measured_search_calls\":" << data.samples.size()
        << ",\"measured_topk_bit_checks\":" << data.samples.size() << ",\n"
        << "\"measured_debug_download_calls\":0,\"measured_poison_calls\":0,\"measured_file_io_calls\":0,\n"
        << "\"diagnostic_checkpoints_outside_measurement\":true,\"host_source_sha256\":{";
    for (size_t i=0;i<BM25_BENCH_SOURCE_COUNT;++i)
        out << (i?",":"") << "\"" << BM25_BENCH_SOURCES[i][0] << "\":\"" << BM25_BENCH_SOURCES[i][1] << "\"";
    out << "},\n\"samples\":[";
    for (size_t i=0;i<data.samples.size();++i) {
        const auto &sample=data.samples[i];
        out << (i?",":"") << "{\"cycle\":" << sample.cycle << ",\"case\":\"" << (sample.which?"B":"A")
            << "\",\"call_wall_ms\":" << sample.callMs << ",\"internal_search_ms\":" << sample.internalMs
            << ",\"query_h2d_bytes\":" << sample.queryH2DBytes << ",\"topk_d2h_bytes\":" << sample.topkD2HBytes << "}";
    }
    out << "],\n"; // Backend-specific fields follow; caller closes JSON and checks stream.
}
}
