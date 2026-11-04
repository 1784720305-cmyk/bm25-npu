#include "common/bm25/error.h"
#include "common/bm25/fixed_query_packet.h"
#include "common/bm25/validation.h"
#include <exception>
#include <iostream>
#include <vector>

int main()
{
    try {
        const bm25plan::PacketLayout layout(1, 2, 1);
        std::vector<unsigned char> arena(layout.inputBytes, 0xa7);
        const auto before = arena;
        bool rejected = false;
        try {
            layout.Pack({{1, -1}}, arena.data());
        } catch (const ascend::bm25::InvalidArgument &) {
            rejected = true;
        }
        ascend::bm25::Check(rejected && arena == before, "Invalid request modified reusable query memory");
        ascend::bm25::Check(layout.Pack({{1, 2}}, arena.data()) == 24, "Valid request after rejection failed");
        const ascend::bm25::DeviceError error(161001, "prepare failed");
        ascend::bm25::Check(error.GetCode() == 161001 && std::string(error.what()) == "prepare failed",
                            "Device error lost code or diagnostic");
        std::cout << "PASS: typed input rejection, unchanged packet and subsequent valid packing\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
