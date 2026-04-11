#include "../third_party/incbin.h"
#include "policy_network.hpp"
#include "value_network.hpp"

#include <cstdlib>
#include <iostream>

namespace network {

namespace detail {

template <typename Network>
const Network *checked_network(const unsigned char *data, i64 size, const char *name) {
    constexpr usize MAX_PADDING = 128;
    const i64 diff = size - sizeof(Network);
    if (std::abs(diff) > MAX_PADDING) {
        std::cerr << "network size mismatch for " << name << ": file has " << size << " bytes, expected "
                  << sizeof(Network) << " bytes\n";
        std::exit(1);
    }
    return reinterpret_cast<const Network *>(data);
}

} // namespace detail

#ifdef EVALFILE

namespace detail {

struct CombinedNetworks {
    value::ValueNetwork value_net;
    alignas(64) policy::PolicyNetwork policy_net;
};
INCBIN(COMBINEDNETWORKS, EVALFILE);
const auto combined_networks =
    checked_network<CombinedNetworks>(gCOMBINEDNETWORKSData, gCOMBINEDNETWORKSSize, "combined network");

} // namespace detail

namespace value {

extern const auto network = &detail::combined_networks->value_net;

} // namespace value

namespace policy {

extern const auto network = &detail::combined_networks->policy_net;

} // namespace policy

#else

namespace value {

INCBIN(VALUENETWORK, VALUEFILE);
extern const auto network = detail::checked_network<ValueNetwork>(gVALUENETWORKData, gVALUENETWORKSize, "value network");

} // namespace value

namespace policy {

INCBIN(POLICYNETWORK, POLICYFILE);
extern const auto network =
    detail::checked_network<PolicyNetwork>(gPOLICYNETWORKData, gPOLICYNETWORKSize, "policy network");
} // namespace policy

#endif

} // namespace network
