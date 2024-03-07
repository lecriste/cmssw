/**
 Description: Utility function to calculate the Magnetic Field on the GPU
*/

#ifndef DataFormats_EgammaReco_interface_alpaka_MagneticFieldParabolicPortable_h
#define DataFormats_EgammaReco_interface_alpaka_MagneticFieldParabolicPortable_h

#include <Eigen/Core>

using Vector3f = Eigen::Matrix<float, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace MagneticFieldParabolicPortable {

    struct Parameters{
        float c1 = 3.8114;  
        float b0 = -3.94991e-06;   
        float b1 = 7.53701e-06;
        float a = 2.43878e-11; 
    };

    template <typename V3>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float Kr(V3 vec) {
        Parameters p;
        return p.a * (vec(0)*vec(0)+vec(1)*vec(1)) + 1.;
    }

    template <typename V3>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float B0Z(V3 vec) {
        Parameters p;
        return p.b0 * vec(2)*vec(2) + p.b1 * vec(2) + p.c1;
    }

    template <typename V3>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float MagneticFieldAtPoint(V3 vec) {
        Parameters p;
        return B0Z(vec) * Kr(vec);
    }

  } // MagneticFieldParabolicPortable

} // ALPAKA_ACCELERATOR_NAMESPACE

#endif


