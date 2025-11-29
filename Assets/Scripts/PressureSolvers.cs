using UnityEngine;

public class PressureSolvers
{
    //Exponetial equation of state
    public static float TaitPressure(float rho, float rho0, float c0, float gamma = 7f)
    {
        float B = c0 * c0 * rho0 / gamma;
        return B * (Mathf.Pow(rho / rho0, gamma) - 1f);
    }
    //Linear equation of state
    public static float LinearEOS(float rho, float rho0, float k)
    {
        // k = stiffness constant
        return k * (rho - rho0);
    }

}
