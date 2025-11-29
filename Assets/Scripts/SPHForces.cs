using SPH;
using System.Runtime.CompilerServices;
using Unity.Mathematics;
using UnityEngine;


public class SPHForces
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float poly6Density( in SPH.SPHParams sph,float mass, float d2)
    {
        return mass * Kernels_SPH.poly6(sph.h2, d2, sph.poly6Coeff);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float wendlandC2Density(in SPH.SPHParams sph, float mass, float d2)
    {
        return mass * Kernels_SPH.WendlandC2(sph.h,sph.h2, d2, sph.wendlandC2Coeff);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float3 SymetricMonaghanViscosity(in SPH.SPHParams sph, in SPH.PhysicsParams physParam
        , float d2,float3 v_ji,float rho_j,float rho_i)
    {

        float rlen = math.sqrt(d2);
        float lapW = Kernels_SPH.ViscLaplacian(sph.h,sph.viscLapCoeff, rlen);
        float invRhoProd = 1f/(rho_j * rho_i);
        float avgInvRho = 2f / (rho_i + rho_j);

        return physParam.viscosity * physParam.mass * invRhoProd * (v_ji) * lapW;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float3 PressureForceWithArtificialViscosity(in SPH.SPHParams sph, in SPH.PhysicsParams physParam,
        float3 r_ij, float p_i,float invRho_i,float p_j
        , float invRho_j,float vr,float d2)
    {
     

        float3 gradW = Kernels_SPH.WendlandC2Grad(sph.h, sph.h2, r_ij,d2, sph.wendlandC2Coeff);
        float gradMag = math.length(gradW);
      
        // Symmetric pressure
        float coeffP = physParam.mass * (
            p_j * invRho_j * invRho_j +
            p_i * invRho_i * invRho_i
        );


        // Artificial viscosity (Monaghan)
        float Pi_ij = 0f;
        if (vr < 0f)
        {
            float mu = sph.h * vr / (d2 + physParam.avEps * sph.h2);
            float invRhoAvg = 0.5f * (invRho_i + invRho_j);
            Pi_ij = (-physParam.avAlpha * physParam.soundSpeed * mu + physParam.avBeta * mu * mu) * invRhoAvg;
        }

        float coeff = coeffP + physParam.mass * Pi_ij;
        return -coeff * gradW;
    }

    public static float3 xsphCorrection(in SPHParams sph, in PhysicsParams phys,float d2,float3 v_ji,
        float rho_i,float rho_j)
    {
        float Wij = Kernels_SPH.WendlandC2(sph.h,sph.h2, d2, sph.wendlandC2Coeff);
        if (Wij >= 1e-4f)
        {
            float w = phys.mass / (0.5f * (rho_i + rho_j));
            return  w * (v_ji) * Wij;
        }
        return 0;
    }
 

}

