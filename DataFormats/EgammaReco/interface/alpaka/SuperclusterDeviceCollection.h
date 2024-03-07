#ifndef DataFormats_EgammaReco_interface_alpaka_SuperclusterDeviceCollection_h
#define DataFormats_EgammaReco_interface_alpaka_SuperclusterDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/EgammaReco/interface/SuperClusterSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace reco {

    // make the names from the top-level portabletest namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::portabletest namespace
    using namespace ::reco;

    // SoA with x, y, z, id fields in device global memory
    using SuperclusterDeviceCollection = PortableCollection<SuperClusterSoA>;

  }  // namespace portabletest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_PortableTestObjects_interface_alpaka_TestDeviceCollection_h
