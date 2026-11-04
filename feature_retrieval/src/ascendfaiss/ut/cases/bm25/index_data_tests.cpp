#include "common/bm25/index_data.h"
#include "common/bm25/validation.h"
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <limits>

using ascend::bm25::Bytes;
using ascend::bm25::Check;
using ascend::bm25::Index;

template <class Function> void Reject(Function function)
{
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    Check(rejected, "Invalid memory index accepted");
}

int main()
{
    try {
        const Index valid({0, 2}, {0, 2}, {1, 3}, {1, 2, 1}, {0.5F});
        Check(valid.docs.size() == 2 && valid.norm.size() == 3, "Memory index data changed");
        const Index emptyPostings({0, 0}, {}, {}, {1}, {0});
        Check(emptyPostings.docs.empty(), "Empty posting list must be supported");
        Reject([] { Index({}, {}, {}, {}, {}); });
        Reject([] { Index({0, 1}, {0}, {1}, {}, {1}); });
        Reject([] { Index({0, 2}, {0, 0}, {1, 1}, {1}, {1}); });
        Reject([] { Index({0, 2}, {1, 0}, {1, 1}, {1, 1}, {1}); });
        Reject([] { Index({0, 1}, {1}, {1}, {1}, {1}); });
        Reject([] { Index({0, 1}, {0}, {1.5F}, {1}, {1}); });
        Reject([] { Index({0, 0}, {}, {}, {0}, {1}); });
        Reject([] { Index({0, 0}, {}, {}, {1}, {-1}); });
        Reject([] { Index({0, 0}, {}, {}, {1}, {std::numeric_limits<float>::quiet_NaN()}); });
        Reject([] { Index({1, 1}, {}, {}, {1}, {1}); });
        Reject([] { Index({0, 1, 0}, {}, {}, {1}, {1, 1}); });
        Reject([] { Bytes(1, 0); });
        Reject([] { Bytes(UINT64_MAX, 8); });
        Check(Bytes(0, 4) == 0 && Bytes(5, 4) == 20, "Byte sizing changed");
        std::cout << "PASS: in-memory index validation and checked byte sizes\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
