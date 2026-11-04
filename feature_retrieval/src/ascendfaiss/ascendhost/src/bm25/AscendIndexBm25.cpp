#include "ascendhost/include/index/AscendIndexBm25.h"
#include "ascendhost/include/impl/Bm25ErrorAdapter.h"
#include "ascendhost/src/bm25/large_search.h"
#include "common/bm25/index_data.h"
#include "common/bm25/error_info.h"
#include <memory>
#include "common/ErrorCode.h"
#include "common/bm25/error.h"
#include "common/bm25/search_inputs.h"
#include <mutex>
#include <utility>

namespace faiss::ascend {
class AscendIndexBm25::Impl {
public:
    struct State {
        ::ascend::bm25::Index index;
        ::bm25large::Session session;
        State(::ascend::bm25::Index source, const Bm25Config &config)
            : index(std::move(source)), session(index, config.vendorDirectory, config.deviceId, config.queryCount,
                                                config.topK, config.maxQueryWords, {config.context, config.resources})
        {
        }
    };
    mutable std::mutex mutex;
    ::ascend::bm25::ErrorInfo error;
    std::unique_ptr<State> state;
};

AscendIndexBm25::AscendIndexBm25() : impl_(std::make_unique<Impl>()) {}
AscendIndexBm25::~AscendIndexBm25() = default;

::ascend::APP_ERROR AscendIndexBm25::Init(const ::ascend::bm25::Index &index, const Bm25Config &config)
{
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return ::ascend::bm25::InvokeSdkOperation(
        [&] {
            if (impl_->state) {
                throw ::ascend::bm25::InvalidState("BM25 is already initialized; Close before Init");
            }
            ::ascend::bm25::CheckArgument(config.context && config.resources && !config.vendorDirectory.empty(),
                                          "SDK context, resources and operator vendors are required");
            // Index fields are public data; revalidate the snapshot in case the caller mutated it.
            ::ascend::bm25::Index checked(index.offsets, index.docs, index.tf, index.norm, index.idf);
            impl_->state = std::make_unique<Impl::State>(std::move(checked), config);
        },
        &impl_->error);
}

::ascend::APP_ERROR AscendIndexBm25::Search(const ::bm25search::Queries &queries, Bm25SearchResult &result)
{
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return ::ascend::bm25::InvokeSdkOperation(
        [&] {
            if (!impl_->state) {
                throw ::ascend::bm25::InvalidState("BM25 is not initialized");
            }
            auto selected = impl_->state->session.Search(queries);
            Bm25SearchResult next;
            next.labels.assign(selected.ids.begin(), selected.ids.end());
            next.scores = std::move(selected.scores);
            result = std::move(next);
        },
        &impl_->error);
}

::ascend::APP_ERROR AscendIndexBm25::Close()
{
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return ::ascend::bm25::InvokeSdkOperation(
        [&] {
            if (impl_->state) {
                impl_->state->session.Close();
                impl_->state.reset();
            }
        },
        &impl_->error);
}

::ascend::bm25::ErrorInfo AscendIndexBm25::GetLastError() const
{
    std::lock_guard<std::mutex> guard(impl_->mutex);
    return impl_->error;
}
} // namespace faiss::ascend
