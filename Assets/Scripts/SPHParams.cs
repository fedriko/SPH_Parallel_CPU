using Unity.Mathematics;

namespace SPH
{
    public struct SPHParams
    {
        public float h;
        public float h2;

        public float viscLapCoeff;
        public float poly6Coeff;
        public float spikyCoeff;
        public float wendlandC2Coeff;

        public int numberOfParticles;
    }

    public struct PhysicsParams
    {
        public float viscosity;

        public float mass;
        public float invMass;

        public float avAlpha;
        public float avBeta;
        public float avEps;

        public float gamma;
        public float soundSpeed;

        public float pressureStiff;
        public float restDensity;

        public float3 gravity;

    }

    public struct IntegrationParams
    {
        public float dt;
        public float xsphEps;
        public float maxVelocity;
        public bool useXSPH;
        public bool xsphAppliesToPosition;

    }

    public struct CollisionParams
    {
        public float3 boundsSize;
        public float3 boxPosition;
        public float3 velocity;
        public float3 angularVelocity;
        public float collisionRadius;
        public float collisionDamping;
        
    }

    public struct DampingParams
    {
        public bool enabled;
        public float speedThreshold;
        public float densityErrorThreshold;
        public float factor;
    }
}