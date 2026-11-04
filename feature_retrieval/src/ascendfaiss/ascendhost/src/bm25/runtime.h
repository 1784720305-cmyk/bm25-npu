#ifndef BM25_RUNTIME_H
#define BM25_RUNTIME_H
#include "acl/acl_base_rt.h"
#include "ascendhost/include/impl/Bm25RuntimeBinding.h"
#include "common/bm25/error.h"

namespace ascend::bm25 {
void RequireAcl(aclError code, const char *operation);

class Runtime {
public:
    Runtime() = default;
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;
    ~Runtime();
    void Initialize(int deviceId, const RuntimeBinding &binding = {});
    bool HasCurrentContext() const noexcept;
    void RequireCurrentContext() const;
    bool Close() noexcept;
    void Abandon() noexcept;

    // Also used by standalone diagnostic executables that initialize ACL in stages.
    bool initialized = false;
    bool selected = false;
    int device = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

private:
    bool borrowedContext_ = false;
    // Allocated before acquiring device resources. On unsafe teardown, deliberately
    // retain this lease together with the device handles; no allocation in destructors.
    std::unique_ptr<std::shared_ptr<void>> ownerLease_;
};
} // namespace ascend::bm25
#endif // BM25_RUNTIME_H
