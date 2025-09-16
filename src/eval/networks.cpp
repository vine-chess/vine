#include "../third_party/incbin.h"
#include "../util/assert.hpp"
#include "../util/simd.hpp"
#include "policy_network.hpp"
#include "value_network.hpp"

namespace network {

#ifdef EVALFILE

namespace detail {

struct CombinedNetworks {
    value::ValueNetwork value_net;
    alignas(64) policy::PolicyNetwork policy_net;
};
INCBIN(COMBINEDNETWORKS, EVALFILE);
const auto combined_networks = []() {
    const auto alignment = alignof(CombinedNetworks);
    vine_assert(util::next_multiple(gCOMBINEDNETWORKSSize, alignment) == sizeof(CombinedNetworks));
    return reinterpret_cast<const CombinedNetworks *>(gCOMBINEDNETWORKSData);
}();

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
extern const auto network = []() {
    const auto alignment = alignof(ValueNetwork);
    vine_assert(util::next_multiple(gVALUENETWORKSize, alignment) == sizeof(ValueNetwork));
    return reinterpret_cast<const ValueNetwork *>(gVALUENETWORKData);
}();

} // namespace value

namespace policy {

INCBIN(POLICYNETWORK, POLICYFILE);
extern const auto network = []() {
    const auto alignment = alignof(PolicyNetwork);
    vine_assert(util::next_multiple(gPOLICYNETWORKSize, alignment) == sizeof(PolicyNetwork));
    return reinterpret_cast<const PolicyNetwork *>(gPOLICYNETWORKData);
}();
} // namespace policy

#endif

} // namespace network
