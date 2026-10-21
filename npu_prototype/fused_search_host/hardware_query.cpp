// Read-only capability query; no kernel launch, no device reset, no compiler flag changes.
#include "acl/acl.h"
#include <cstdlib>
#include <iostream>
int main(int argc,char **argv)
{
    if (argc!=2) return 2;
    const int device=std::atoi(argv[1]);
    if (device<0 || aclInit(nullptr)!=ACL_ERROR_NONE) return 1;
    const aclrtDevAttr attrs[]={ACL_DEV_ATTR_VECTOR_CORE_NUM,ACL_DEV_ATTR_WARP_SIZE,
        ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE,ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE};
    const char *names[]={"aiv_core_count","warp_threads","max_resident_threads_per_aiv","ub_bytes_per_aiv"};
    for (unsigned i=0;i<4;++i) {
        int64_t value=0;const auto status=aclrtGetDeviceInfo(device,attrs[i],&value);
        std::cout<<names[i]<<" status="<<status<<" value="<<value<<'\n';
    }
    std::cout<<"scope=runtime capability query; not tiling resource restriction, occupancy or actual register usage\n";
    return aclFinalize()==ACL_ERROR_NONE?0:1;
}
