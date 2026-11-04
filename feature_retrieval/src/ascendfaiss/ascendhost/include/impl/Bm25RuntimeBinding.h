#ifndef BM25_RUNTIME_BINDING_H
#define BM25_RUNTIME_BINDING_H
#include "acl/acl_base_rt.h"
#include <memory>

namespace ascend::bm25 {
// The caller activates this context before construction, Search and Close, and keeps
// it alive. owner retains the SDK resource proxy until all BM25 resources are freed.
struct RuntimeBinding {
    aclrtContext context = nullptr;
    std::shared_ptr<void> owner;
};
} // namespace ascend::bm25
#endif // BM25_RUNTIME_BINDING_H
