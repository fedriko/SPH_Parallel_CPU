using System.Runtime.CompilerServices;
using Unity.Mathematics;
using UnityEngine;

public static class Kernels_SPH
{

    // Kernel used for densities 
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float poly6(float h2, float3 r, float poly6Coeff)
    {
        float r2 = math.lengthsq(r);

        if (r2 >= h2)
            return 0f;

        float h2_minus_r2 = h2 - r2;
        return poly6Coeff * h2_minus_r2 * h2_minus_r2 * h2_minus_r2;


    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    //Kernel used for pressure. Prevents clumping of particles.
    //Strong repulsive forces
    public static float3 SpikyGrad3D(float h, float h2, float3 r, float spikyCoeff)
    {
        float d2 = math.lengthsq(r);
        if (d2 <= 1e-12f || d2 >= h2)
            return float3.zero;

        float rlen = Mathf.Sqrt(d2);
        float x = h - rlen;
        float coeff = spikyCoeff * (x * x) / rlen;

        return coeff * r;
    }


    // Opperators (Laplacian for viscosity)
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float ViscLaplacian(float h, float viscLapCoeff, float rlen)
    {
        if (rlen >= h) return 0f;
        return viscLapCoeff * (h - rlen);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float WendlandC2(float h, float h2, float d2 ,float wendlandC2Coeff)
    {
        if (d2>= h2)
            return 0f;

        float rlen = math.sqrt(d2);
        float q = rlen / h;
        float oneMinusQ = 1f - q;
        float oneMinusQ4 = oneMinusQ * oneMinusQ * oneMinusQ * oneMinusQ;
        return wendlandC2Coeff * oneMinusQ4 * (1f + 4f * q);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float3 WendlandC2Grad(float h, float h2, float3 r,float d2, float wendlandC2Coeff)
    {

        if (d2 >= h2)
            return float3.zero;

        float rlen = math.sqrt(d2);
        float q = rlen / h;

        float oneMinusQ = 1f - q;
        float oneMinusQ3 = oneMinusQ * oneMinusQ * oneMinusQ;


        float coeff = -20f * (wendlandC2Coeff / h)* oneMinusQ3 * (q / rlen);

        return coeff * r;
    }
}
