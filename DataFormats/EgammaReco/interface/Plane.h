#ifndef DataFormats_EgammaReco_interface_Plane_h
#define DataFormats_EgammaReco_interface_Plane_h

#include <Eigen/Dense>
#include <cmath>

namespace PlanePortable {

    template <typename Vec3>
    struct Plane {
        Vec3 position;
        Vec3 rotation; // This is rotation().z()

        constexpr Vec3 pos() const {
            return position;
        }

        // Returns the normal vector of the plane
        constexpr Vec3 normalVector() const {
            return rotation;
        }

        // Fast access to distance from plane for a point
        constexpr float localZ(const Vec3& vp) const {
            return normalVector().dot(vp - position);
        }

        // Clamped distance from plane for a point
        constexpr float localZclamped(const Vec3& vp) const {
            auto d = localZ(vp);
            return std::abs(d) > 1e-7f ? d : 0; 
        }

        // Fast access to distance from plane for a vector
        constexpr float distanceFromPlaneVector(const Vec3& gv) const {
            printf("\ninside distanceFromPlaneVector, normalVector : (%f, %f, %f)", normalVector().x(), normalVector().y(), normalVector().z());
            return normalVector().dot(gv);
        }
    };

}  // namespace PlanePortable

#endif // DataFormats_EgammaReco_interface_Plane_h