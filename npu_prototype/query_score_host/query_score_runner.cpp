// Confirmed generated ACLNN API. Score-only A/B/A session diagnostic; no CPU score calculation.
#include "query_score_runtime.h"
#include <cctype>
#include <sstream>

#if !defined(BM25_CMAKE_SHA256) || !defined(BM25_RUNTIME_SHA256) || !defined(BM25_RUNNER_SHA256)
#error Build this diagnostic through its CMakeLists.txt to record the compiled source fingerprints.
#endif

struct Index {
    std::vector<int64_t> offsets;
    std::vector<int32_t> docs;
    std::vector<float> tf, norm, idf;
    explicit Index(const fs::path &directory)
        : offsets(Read<int64_t>(directory/"term_offsets.bin")), docs(Read<int32_t>(directory/"posting_doc_ids.bin")),
          tf(Read<float>(directory/"posting_tf.bin")), norm(Read<float>(directory/"doc_norm.bin")),
          idf(Read<float>(directory/"idf.bin"))
    {
        Check(offsets.size() == idf.size()+1 && docs.size() == tf.size() &&
              offsets.front() == 0 && offsets.back() == static_cast<int64_t>(docs.size()), "Index dimensions/endpoints differ");
        for (float value : norm) Check(std::isfinite(value) && value > 0,"Invalid article norm");
        for (size_t term = 0; term < idf.size(); ++term) {
            Check(std::isfinite(idf[term]) && idf[term] >= 0,"Invalid IDF");
            const int64_t begin = offsets[term], end = offsets[term+1];
            Check(begin >= 0 && begin <= end && end <= static_cast<int64_t>(docs.size()),"Invalid posting bounds");
            for (int64_t pos = begin; pos < end; ++pos) {
                Check(docs[pos] >= 0 && static_cast<size_t>(docs[pos]) < norm.size() &&
                      (pos == begin || docs[pos-1] < docs[pos]),"Unsorted/duplicate/out-of-range posting ID");
                Check(std::isfinite(tf[pos]) && tf[pos] >= 1 && tf[pos] <= 16777216.0F &&
                      std::floor(tf[pos]) == tf[pos],"Invalid posting TF");
            }
        }
    }
};
struct Request {
    std::string label;
    fs::path directory;
    std::vector<int64_t> offsets;
    std::vector<int32_t> terms;
    std::vector<float> expected, actual;
    uint64_t workspace = 0;
    bool repeated = false;
    int64_t Queries() const { return static_cast<int64_t>(offsets.size()-1); }
};

std::vector<Request> LoadRequests(const fs::path &plan, size_t docs)
{
    std::ifstream file(plan);
    Check(file.is_open(),"Cannot read request plan");
    std::vector<Request> result;
    std::map<std::string,bool> labels;
    std::string line;
    uint64_t totalScores = 0;
    while (std::getline(file,line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto separator = line.find('\t');
        Check(separator != std::string::npos && separator > 0 && line.find('\t',separator+1) == std::string::npos,
              "Request plan needs label<TAB>query-directory");
        Request request;
        request.label = line.substr(0,separator);
        Check(std::all_of(request.label.begin(),request.label.end(),[](unsigned char c){return std::isalnum(c) || c == '_';}) &&
              labels.emplace(request.label,true).second,"Unsafe/duplicate request label");
        request.directory = fs::canonical(line.substr(separator+1));
        request.offsets = Read<int64_t>(request.directory/"query_offsets.bin");
        request.terms = Read<int32_t>(request.directory/"query_term_ids.bin",true);
        Check(request.offsets.size() >= 2 && request.offsets.front() == 0 &&
              request.offsets.back() == static_cast<int64_t>(request.terms.size()),"Bad query boundary endpoints");
        for (size_t i = 1; i < request.offsets.size(); ++i)
            Check(0 <= request.offsets[i-1] && request.offsets[i-1] <= request.offsets[i],"Nonmonotonic query boundaries");
        for (int32_t word : request.terms) Check(word >= 0 && word < INT32_MAX,"Bad query ID");
        Check(static_cast<uint64_t>(request.Queries()) <= static_cast<uint64_t>(INT32_MAX)/docs,"Q*N exceeds operator limit");
        const uint64_t count = static_cast<uint64_t>(request.Queries())*docs;
        Check(count <= 16777216 && totalScores <= 16777216-count,"Diagnostic total score budget exceeded");
        totalScores += count;
        request.expected = Read<float>(request.directory/"expected_scores.bin");
        Check(request.expected.size() == count,"Reference dimensions differ");
        for (float value : request.expected) Check(std::isfinite(value) && value >= 0,"Invalid reference value");
        result.push_back(std::move(request));
        Check(result.size() <= 32,"Diagnostic plan exceeds 32 requests");
    }
    Check(file.eof() && !result.empty(),"Empty/unreadable request plan");
    return result;
}

void CompareReference(const Request &request)
{
    Check(request.actual.size() == request.expected.size(),"Output dimensions differ");
    for (size_t i = 0; i < request.actual.size(); ++i) {
        const double a = request.actual[i], e = request.expected[i];
        if (!std::isfinite(a) || a < 0 || std::abs(a-e) > 1e-6+2e-6*std::abs(e)) {
            std::ostringstream message;
            message.precision(17);
            message << "Device/reference mismatch: " << request.label << " element=" << i
                    << " actual=" << a << " expected=" << e << " tolerance=" << 1e-6+2e-6*std::abs(e);
            throw std::runtime_error(message.str());
        }
    }
}

void SaveReport(const fs::path &directory, const std::vector<Request> &requests, uint64_t indexBytes, int device)
{
    size_t largestOutputBytes = 0;
    for (const auto &r : requests) largestOutputBytes = std::max(largestOutputBytes,Bytes(r.actual.size(),4));
    std::ofstream file(directory/"device_session.json");
    file << "{\n  \"status\": \"device_score_session_completed_reference_checked\",\n"
         << "  \"device\": " << device << ",\n  \"index_upload_batches\": 1,\n"
         << "  \"index_tensor_h2d_calls\": 5,\n  \"index_upload_bytes\": " << indexBytes << ",\n"
         << "  \"index_readback_byte_identical\": true,\n  \"index_device_allocations_reused\": true,\n"
         << "  \"debug_index_d2h_calls\": 5,\n  \"debug_index_d2h_bytes\": " << indexBytes << ",\n"
         << "  \"transfer_counter_scope\": \"explicit application aclrtMemcpy calls; framework/internal transfers not profiled\",\n"
         << "  \"scope\": \"score-only diagnostic; includes poison/reference/debug work; no search timing or Top-K\",\n"
         << "  \"host_source_sha256\": {\n"
         << "    \"npu_prototype/query_score_host/CMakeLists.txt\": \"" << BM25_CMAKE_SHA256 << "\",\n"
         << "    \"npu_prototype/query_score_host/query_score_runtime.h\": \"" << BM25_RUNTIME_SHA256 << "\",\n"
         << "    \"npu_prototype/query_score_host/query_score_runner.cpp\": \"" << BM25_RUNNER_SHA256 << "\"\n  },\n"
         << "  \"requests\": [\n";
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto &r = requests[i];
        const size_t offsetBytes = Bytes(r.offsets.size(),8);
        const size_t wordBytes = Bytes(r.terms.size(),4);
        file << (i ? ",\n" : "") << "    {\"label\": \"" << r.label << "\", \"queries\": " << r.Queries()
             << ", \"logical_word_count\": " << r.terms.size()
             << ", \"logical_query_bytes\": " << offsetBytes+wordBytes
             << ", \"query_h2d_bytes\": " << offsetBytes+std::max<size_t>(4,wordBytes)
             << ", \"query_tensor_h2d_calls\": 2, \"score_d2h_bytes\": " << Bytes(r.actual.size(),4)
             << ", \"score_d2h_calls\": 1, \"debug_poison_h2d_calls\": 1, \"debug_poison_h2d_bytes\": " << largestOutputBytes
             << ", \"workspace_bytes\": " << r.workspace << ", \"output_poisoned\": true"
             << ", \"fresh_executor\": true, \"full_reference_checked\": true, \"repeated_query_bits_checked\": "
             << (r.repeated ? "true" : "false") << "}";
    }
    file << "\n  ]\n}\n";
    file.close(); Check(static_cast<bool>(file),"Cannot save device session report");
}

int main(int argc, char **argv)
{
    try {
        if (argc == 3 && std::string(argv[1]) == "--check-library") {
            QueryApi api(fs::absolute(argv[2]));
            std::cout << "Both actual query-score API symbol owners checked. No device work submitted.\n";
            return 0;
        }
        if (argc != 6) {
            std::cerr << "Usage: bm25_query_score_runner INDEX_DIR PLAN_TSV OUTPUT_DIR DEVICE VENDOR_DIR\n";
            return 2;
        }
        static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4 && sizeof(int64_t) == 8);
        static_assert(std::numeric_limits<float>::is_iec559);
        const uint16_t endian = 1;
        Check(*reinterpret_cast<const unsigned char *>(&endian) == 1,"Requires little-endian host");
        const fs::path output = fs::absolute(argv[3]);
        Check(!fs::exists(output),"Output exists; choose a new directory");
        size_t parsed = 0;
        const long number = std::stol(argv[4],&parsed);
        Check(parsed == std::string(argv[4]).size() && number >= 0 && number <= INT32_MAX,"Invalid device");
        Stage("validate_static_index_and_request_plan");
        const Index index(fs::absolute(argv[1]));
        auto requests = LoadRequests(fs::absolute(argv[2]),index.norm.size());
        size_t maxQueries = 0, maxWords = 1;
        for (const auto &request : requests) {
            maxQueries = std::max(maxQueries,static_cast<size_t>(request.Queries()));
            maxWords = std::max(maxWords,request.terms.size());
        }
        QueryApi api(fs::absolute(argv[5]));
        Runtime runtime;
        runtime.device = static_cast<int>(number);
        Stage("runtime_init");
        Require(aclInit(nullptr),"aclInit"); runtime.initialized = true;
        Require(aclrtSetDevice(runtime.device),"aclrtSetDevice"); runtime.selected = true;
        Require(aclrtCreateContext(&runtime.context,runtime.device),"aclrtCreateContext");
        aclrtRunMode mode;
        Require(aclrtGetRunMode(&mode),"aclrtGetRunMode");
        Check(mode == ACL_HOST,"Diagnostic requires ACL_HOST mode");
        Require(aclrtCreateStream(&runtime.stream),"aclrtCreateStream");
        Buffers buffers; buffers.stream = runtime.stream;
        Tensors tensors; tensors.stream = runtime.stream; // Destroy descriptors before memory.
        const void *host[5] = {index.offsets.data(),index.docs.data(),index.tf.data(),index.norm.data(),index.idf.data()};
        const size_t counts[5] = {index.offsets.size(),index.docs.size(),index.tf.size(),index.norm.size(),index.idf.size()};
        const size_t widths[5] = {8,4,4,4,4};
        const aclDataType types[5] = {ACL_INT64,ACL_INT32,ACL_FLOAT,ACL_FLOAT,ACL_FLOAT};
        void *memory[5]; aclTensor *inputs[5];
        uint64_t indexBytes = 0;
        Stage("allocate_and_upload_index_once");
        for (size_t i = 0; i < 5; ++i) {
            const size_t bytes = Bytes(counts[i],widths[i]);
            indexBytes += bytes;
            memory[i] = buffers.Allocate(bytes);
            inputs[i] = tensors.Describe1D(static_cast<int64_t>(counts[i]),types[i],memory[i]);
            Require(aclrtMemcpy(memory[i],bytes,host[i],bytes,ACL_MEMCPY_HOST_TO_DEVICE),"index upload once");
        }
        void *offsetMemory = buffers.Allocate(Bytes(maxQueries+1,8));
        void *wordMemory = buffers.Allocate(Bytes(maxWords,4));
        const size_t outputCount = maxQueries*index.norm.size(), outputCapacity = Bytes(outputCount,4);
        void *scoreMemory = buffers.Allocate(outputCapacity);
        std::vector<uint32_t> poison(outputCount,0xffffffffU);
        uint64_t workspaceCapacity = 0;
        void *workspace = nullptr;
        std::map<std::string,std::vector<float>> previous;
        for (auto &request : requests) {
            std::cout << "request=" << request.label << " queries=" << request.Queries()
                      << " words=" << request.terms.size() << std::endl;
            Stage("poison_entire_output_capacity");
            Require(aclrtMemcpy(scoreMemory,outputCapacity,poison.data(),outputCapacity,ACL_MEMCPY_HOST_TO_DEVICE),"poison output");
            Tensors current; current.stream = runtime.stream;
            auto *offsetTensor = current.Describe1D(request.offsets.size(),ACL_INT64,offsetMemory);
            auto *wordTensor = current.Describe1D(std::max<size_t>(1,request.terms.size()),ACL_INT32,wordMemory);
            auto *outTensor = current.Describe(request.Queries(),index.norm.size(),ACL_FLOAT,scoreMemory);
            const int32_t dummy = 0;
            const void *words = request.terms.empty() ? static_cast<const void *>(&dummy) : request.terms.data();
            const size_t offsetBytes = Bytes(request.offsets.size(),8), wordBytes = Bytes(std::max<size_t>(1,request.terms.size()),4);
            Stage("upload_query_only");
            Require(aclrtMemcpy(offsetMemory,Bytes(maxQueries+1,8),request.offsets.data(),offsetBytes,ACL_MEMCPY_HOST_TO_DEVICE),"query offsets upload");
            Require(aclrtMemcpy(wordMemory,Bytes(maxWords,4),words,wordBytes,ACL_MEMCPY_HOST_TO_DEVICE),"query words upload");
            aclOpExecutor *executor = nullptr;
            Stage("prepare_query_score");
            Require(api.prepare(inputs[0],inputs[1],inputs[2],inputs[3],inputs[4],offsetTensor,wordTensor,outTensor,
                                &request.workspace,&executor),"query-score prepare");
            Check(executor != nullptr && request.workspace <= std::numeric_limits<size_t>::max(),"Invalid executor/workspace");
            if (previous.empty()) {
                workspaceCapacity = request.workspace;
                workspace = buffers.Allocate(static_cast<size_t>(workspaceCapacity));
            } else Check(request.workspace <= workspaceCapacity,"Workspace grew; stop this fixed-capacity diagnostic");
            Stage("execute_query_score");
            Require(api.execute(workspace,request.workspace,executor,runtime.stream),"query-score execute");
            executor = nullptr; // Ordinary executors are framework-owned and consumed by execution.
            Stage("synchronize_and_download_full_scores");
            Require(aclrtSynchronizeStream(runtime.stream),"query-score synchronize");
            request.actual.resize(request.expected.size());
            const size_t bytes = Bytes(request.actual.size(),4);
            Require(aclrtMemcpy(request.actual.data(),bytes,scoreMemory,bytes,ACL_MEMCPY_DEVICE_TO_HOST),"score download");
            CompareReference(request);
            const auto key = request.directory.string();
            const auto prior = previous.find(key);
            if (prior != previous.end()) {
                Check(prior->second.size() == request.actual.size() &&
                      std::memcmp(prior->second.data(),request.actual.data(),bytes) == 0,"Repeated query score bits changed");
                request.repeated = true;
            } else previous.emplace(key,request.actual);
            Check(current.Close(),"Request tensor cleanup failed");
            std::cout << "request_passed=" << request.label << " all_scores=" << request.actual.size() << std::endl;
        }
        Stage("verify_resident_index_readback_unchanged");
        for (size_t i = 0; i < 5; ++i) {
            const size_t bytes = Bytes(counts[i],widths[i]);
            std::vector<unsigned char> actual(bytes);
            Require(aclrtMemcpy(actual.data(),bytes,memory[i],bytes,ACL_MEMCPY_DEVICE_TO_HOST),"index debug readback");
            Check(std::memcmp(actual.data(),host[i],bytes) == 0,"Resident index was modified");
        }
        Stage("release_resources");
        Check(tensors.Close(),"Static tensor cleanup failed");
        Check(buffers.Close(),"Buffer cleanup failed");
        Check(runtime.Close(),"Runtime cleanup failed");
        Stage("save_checked_outputs");
        Check(fs::create_directories(output),"Cannot create fresh output directory");
        for (const auto &request : requests) Write(output/(request.label+"_scores.bin"),request.actual);
        SaveReport(output,requests,indexBytes,runtime.device);
        std::cout << "index_upload_batches=1 index_tensor_h2d_calls=5 index_upload_bytes=" << indexBytes
                  << " requests=" << requests.size() << " index_readback_byte_identical=true\n"
                  << "Score-only resident session diagnostic completed. No Top-K, search timing or speedup claim.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAILED: " << error.what() << "; no CPU result substituted.\n";
        return 1;
    }
}
