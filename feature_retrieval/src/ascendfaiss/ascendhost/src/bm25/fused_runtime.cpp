#include "ascendhost/src/bm25/fused_runtime.h"
#include "acl/acl_base_rt.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include <iostream>
namespace ascend::bm25 {
bool Tensors::Close()
{
    if (items.empty())
        return true;
    if (stream && aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
        std::cerr << "cleanup_failed=tensor synchronization; descriptors retained\n";
        return false;
    }
    while (!items.empty()) {
        if (aclDestroyTensor(items.back()) != ACL_ERROR_NONE) {
            std::cerr << "cleanup_failed=aclDestroyTensor\n";
            return false;
        }
        items.pop_back();
    }
    return true;
}

Tensors::~Tensors() { Close(); }

bool Buffers::Close()
{
    if (items.empty())
        return true;
    if (stream && aclrtSynchronizeStream(stream) != ACL_ERROR_NONE) {
        std::cerr << "cleanup_failed=buffer synchronization; memory retained\n";
        return false;
    }
    while (!items.empty()) {
        if (aclrtFree(items.back()) != ACL_ERROR_NONE) {
            std::cerr << "cleanup_failed=aclrtFree\n";
            return false;
        }
        items.pop_back();
    }
    return true;
}

Buffers::~Buffers() { Close(); }

} // namespace ascend::bm25
