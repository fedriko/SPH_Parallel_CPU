using SPH;
using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.SocialPlatforms;
using static Unity.Collections.AllocatorManager;

/// <summary>
/// 3D SPH fluid simulator using:
/// - Weakly compressible SPH 
/// - Optional single–pass or predictor–corrector scheme
/// - Artificial viscosity (Monaghan)
/// - XSPH velocity smoothing 
/// - Uniform grid + CSR neighborhood structure
/// 
/// The goal is to create an interactable fluid that runs in real-time.
///
/// There are two SPH models implemented:
/// 1. A model that computes all forces in a single pass for better performance (ideal for games).
/// 
/// 2. A predictive-corrective model for fluid motion. This model computes viscosity
///    and external forces to predict velocity, then applies pressure forces to correct
///    for fluid compressions.
///
/// We also implement two different state equations to compute densities:
/// * The Tait equation provides a more reactive fluid (exponential response to densities).
///   It is more prone to chaotic behavior but works well using density and velocity damping.
/// * A linear solver that is generally more robust under heavy flows.
/// 
/// For numerical integration, we use the symplectic Euler method. 
/// It is the algorithm of choice for oscillatory systems in physics simulations because, 
/// while it doesn't perfectly conserve energy at every time step, its energy error 
/// remains bounded and oscillates around the true initial energy value over long periods
/// (by preserving the system's symplectic form).
/// Even though the simulation uses dissipative viscosity forces, their overall impact is 
/// relatively low in this low-viscosity fluid simulation. For this reason, symplectic Euler 
/// remains a robust choice for this kind of application.
/// </summary>
public class FluidSPHArticleCPU : MonoBehaviour
{

    /// <summary> Small epsilon used in artificial viscosity denominator. </summary>
    private const float avEps = 0.01f;

    /// <summary>
    /// Clamp range for density ratio rho / rho0 when density clamping is enabled.
    /// </summary>
    private const float minRatio = 0.7f;
    private const float maxRatio = 1.6f;



    // ------------------------------------------------------------------
    // Time stepping
    // ------------------------------------------------------------------


    [Header("Time Stepping")]
    [Tooltip("Base simulation time step used inside the sub-stepping loop.")]
    [SerializeField, Min(1e-4f)]
    private float simDt = 0.004f;

    [Tooltip(
    "Maximum number of simulation substeps the fluid can take per frame.\n" +
    "Higher values let the simulation catch up when FPS drops, keeping motion more stable but using more CPU.\n" +
    "Lower values do fewer physics updates, which can improve FPS when the simulation is the bottleneck, " +
    "but can make the fluid run in slow motion."
    )]
    [SerializeField, Range(1, 20)]
    private int maxSubsteps = 10;

    // Track how much simulation time we still have to do this frame. (debt)
    private float timeAccumulator = 0f;
    private int stepsTaken = 0;

    [SerializeField] bool displacementGridRebuild = false;


    // ------------------------------------------------------------------
    // Physical / SPH model parameters
    // ------------------------------------------------------------------


    [Header("SPH Model")]
    [Tooltip("If true, use single-pass WCSPH (cheaper, slightly less stable). If false, use predictor–corrector scheme.")]
    [SerializeField]
    private SPHMode singlePassSPH = SPHMode.SinglePass;


    [Tooltip("Rest density of the fluid in kg/m^3.")]
    [SerializeField, Min(1f)]
    private float restDensity = 1000f;


    [Tooltip("Gravitational acceleration magnitude (m/s^2).")]
    [SerializeField]
    private float gravity = 9.8f;


    [Header("Physical viscosity")]
    [Tooltip("Dynamic viscosity coefficient (physical viscosity). Higher = thicker fluid.")]
    [SerializeField, Min(0f)]
    private float viscosity = 0.001f;


    [Header("Artificial Viscosity (Monaghan)")]
    [Tooltip("Coefficient α for Monaghan artificial viscosityparticleSpacing. 0 = disabled.")]
    [SerializeField, Range(0f, 1f)]
    private float avAlpha = 0.03f;

    // β is derived from α 
    private float avBeta;

    [Header("Equation of State (EOS)")]
    [SerializeField]
    private EOS eosType = EOS.Tait;

    [Header("Tait pressure solver")]
    [Tooltip(
        "Gamma parameter used in the Tait equation to compute pressures.\n" +
        "Controls how sharply pressure increases for density deviations from the rest density.\n" +
        "Higher values make the fluid behave stiffer (exponential pressure response), " +
        "but can also increase numerical instability."
    )]
    [SerializeField, Min(1f)]
    private float gamma = 7f;

    [Tooltip("Artificial speed of sound used in the Tait equation of state.\n" +
"Higher values make the fluid effectively stiffer and reduce density error by allowing pressure waves to propagate faster.")]
    [SerializeField, Min(0.1f)]
    private float soundSpeed = 20f;


    [Header("Linear pressure solver")]
    [Tooltip("Stiffness constant for the linear EOS (when linear is true).")]
    [SerializeField]
    private float pressureStiff = 200f;


    [Header("XSPH (Velocity Smoothing)")]
    [Tooltip(
    "If true, enable XSPH velocity smoothing.\n" +
    "XSPH blends each particle's velocity with the velocities of its neighbors,\n" +
    "producing a smoother, more coherent fluid motion."
)]
    [SerializeField]
    private bool useXSph = true;
    [SerializeField]


    private bool xsphAppliesToPosition = false;

    [Tooltip("XSPH coefficient. Higher values blend more with neighbor velocities.")]
    [SerializeField, Range(0f, 1f)]
    private float xsphEps = 0.5f;

    [Header("Stability")]
    [Tooltip("Maximum allowed particle speed. Velocities are clamped to this magnitude.")]
    [SerializeField, Min(0.1f)]
    private float maxVelocity = 30f;

    [Tooltip("Clamp densities to [minRatio, maxRatio] * rest density before computing pressure.")]
    [SerializeField]
    private bool densityClamping = true;

    [Header("Damping")]
    [Tooltip("If true, apply a special damping when both velocity and density error are close to rest.")]
    [SerializeField]
    private bool densityDamping = true;

    [SerializeField, Range(0f, 0.1f)]
    private float densityErrorThreshold = 0.01f;

    [Tooltip("Below this speed, damping is applied to let particles come to rest.")]
    [SerializeField, Range(0f, 1f)]
    private float speedThreshold = 0.1f;

    [Tooltip("Damping factor applied (0 = kill motion, 1 = no damping).")]
    [SerializeField, Range(0f, 1f)]
    private float restDampingFactor = 0.95f;




    [Header("Particle Layout")]
    [Tooltip(
    "Distance between particles in the initial layout.\n" +
    "Combined with eta, it determines the smoothing length: h = eta * particleSpacing.\n" +
    "Smaller spacing = more particles, higher resolution, and higher CPU cost."
)]
    [SerializeField, Min(1e-4f)]
    private float particleSpacing = 0.1f;

    [Tooltip("Vertical offset of the initial particle block relative to the box center.")]
    [SerializeField]
    private float height = 0f;

    [Tooltip(
      "Smoothing-length multiplier: h = eta * particleSpacing.\n" +
      "Eta controls the support radius of the kernel (h) and therefore the number of neighbors.\n" +
      "Larger eta equals more neighbors (smoother, more stable, more expensive!!). Smaller eta equals fewer neighbors (noisier, possibly unstable !!)."
  )]
    [SerializeField, Range(1.6f, 3.2f)]
    private float eta = 2f;

    [Tooltip("Number of particles per axis in the initial cubic block.")]
    [SerializeField, Range(1, 128)]
    private int particlesPerAxis = 23;



    // ------------------------------------------------------------------
    // Collision / container
    // ------------------------------------------------------------------

    [Header("Container / Collisions")]
    [Tooltip("Damping of velocity on collision with box walls (0 = perfectly elastic, 1 = fully inelastic).")]
    [SerializeField, Range(0f, 1f)]
    private float collisionDamping = 0.3f;

    [Tooltip("Size of the box in local space (scaled and rotated by this GameObject).")]
    [SerializeField]
    private float3 boundsSize = new float3(3.3f, 3.3f, 3.3f);

    [Tooltip("Jitter applied to initial particle positions (as a fraction of spacing). 0 = no jitter.")]
    [SerializeField, Range(0f, 1f)]
    private float jitterStrength = 0f;

    // ------------------------------------------------------------------
    // Rendering
    // ------------------------------------------------------------------

    [Header("Rendering")]
    [Tooltip("Scale factor applied to the rendered sphere radius relative to particle spacing.")]
    [SerializeField, Range(0.01f, 2f)]
    private float ResizeFactor = 0.7f;

    [Tooltip("Resolution used for the generated sphere mesh.")]
    [SerializeField, Range(1, 8)]
    private int resolution = 4;

    [Tooltip("Material used for instanced particle rendering.")]
    [SerializeField]
    private Material material;

    // ------------------------------------------------------------------
    // Internal physical quantities (computed at runtime)
    // ------------------------------------------------------------------

    private float mass;          // Particle mass
    private float radius;        // Render radius
    private SPHParams sphParams;
    private PhysicsParams physicsParams;
    private CollisionParams collisionParams;
    private DampingParams dampingParams;
    private IntegrationParams intParams;

    // Arrays storing particle state & intermediate data (temps)
    private NativeArray<float3> forces;
    private NativeArray<float3> pressureForce_tmp;
    private NativeArray<float3> positions;
    private NativeArray<float3> positions_prev;
    private NativeArray<float3> positions_tmp;
    private NativeArray<float3> velocities;
    private NativeArray<float3> velocities_tmp;
    private NativeArray<float3> velocities_saved;
    private NativeArray<float3> velocities_old_tmp;
    private NativeArray<float> densities;
    private NativeArray<float> densities_tmp;
    private NativeArray<float> pressures;
    private NativeArray<float> pressures_tmp;
    private NativeArray<float> invDensities;
    private NativeArray<float> invDensities_tmp;

    private int numberOfParticles;

    private Mesh sphereMesh;
    private RenderParams rp;
    private float collisionRadius;

    private Matrix4x4[] mats;

    // Kernel parameters (computed once per frame)
    private float h;
    private float h2;

    // Box bounds
    private Vector3 centerBox;
    private float3x3 rotationMatrix;
    private float3x3 rotationMatrixInverse;

    //CSR grid
    public CSR_Grid grid;


    private NativeArray<float3> xsphDelta;
    private NativeArray<float3> xsphDelta_tmp;

    private JobHandle handle;

    //BoxAngles and Velocity
    private Quaternion qPrevBox = Quaternion.identity;
    private float3 boxPrevPos = 0;

    //Max velocity
    private NativeReference<float> maxVel, accumulatedDisplacement;
    private float deltaH;
    private NativeReference<bool> rebuildGrid;
    private NativeArray<float> perThreadMaxDispSq;

    void Awake()
    {


        // Remove vSync / FPS cap so the sim is not throttled artificially.
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = -1;

        // Precompute all constant parameters and kernels that depend on user settings.
        CalculateParameters();

        // Set up rendering (RenderParams + sphere mesh).
        CreateRp();

        // Allocate and initialize all native arrays + particle positions.
        InitialiseArraysAndPositions();

        // Uniformly scale the mesh so its radius matches our rendering radius.
        ScaleMesh(sphereMesh);


        BuildGridAndNbr();

    }


    private void Update()
    {

        //Update prams
        UpdateParams();

        //We add this frame to the accumulator
        timeAccumulator += Time.deltaTime;




        //We put a safe limit to time accumulator 
        //We cant bank more than the maxSubsteps
        float maxAccumulation = simDt * maxSubsteps;
        if (timeAccumulator > maxAccumulation)
        {
            timeAccumulator = maxAccumulation;
        }

        stepsTaken = 0;


        while (timeAccumulator >= simDt && stepsTaken < maxSubsteps)
        {
            StepOnce(simDt);
            timeAccumulator -= simDt;
            stepsTaken++;
        }


        for (int i = 0; i < numberOfParticles; i++)
            mats[i] = Matrix4x4.Translate(positions[i]);

        //We draw spheres in batch using RenderMeshInstanced
        const int BATCH = 1023;
        for (int start = 0; start < numberOfParticles;)
        {
            int count = Mathf.Min(BATCH, numberOfParticles - start);
            Graphics.RenderMeshInstanced<Matrix4x4>(rp, sphereMesh, 0, mats, count, start);
            start += count;
        }



    }

    /// <summary>
    /// Performs a single SPH simulation step of duration dt.
    /// Chooses between single-pass and predictor–corrector WCSPH.
    /// </summary>
    void StepOnce(float dt)
    {
        if (rebuildGrid.Value || !displacementGridRebuild)
        {
            BuildGridAndNbr();
            rebuildGrid.Value = false;
            accumulatedDisplacement.Value = 0f;
        }
        switch (singlePassSPH)
        {
            case SPHMode.SinglePass:
                singlePassWCSPH(dt);
                break;

            case SPHMode.PredictiveCorrective:
                Predictive_Corrective_WCSPH(dt);
                break;
        }
    }

    /// <summary>
    /// Single-pass WCSPH:
    /// 1. Build spatial grid + neighbor CSR.
    /// 2. Compute densities.
    /// 3. Compute pressures.
    /// 4. Compute all forces (pressure + viscosity + gravity + XSPH).
    /// 5. Integrate and handle collisions using symplectic euler method (energy preservation).
    /// </summary>
    private void singlePassWCSPH(float dt)
    {
        // 1) Update box transform for collision detection
        centerBox = transform.position;
        rotationMatrix = new float3x3(transform.rotation);
        rotationMatrixInverse = math.transpose(rotationMatrix);


        // 4) Compute densities
        CalculateDensities D_job = new CalculateDensities
        {
            physicsParams = physicsParams,
            sphParams = sphParams,
            densities = this.densities,
            nbrStart = grid.NbrStart,
            nbrEnd = grid.NbrEnd,
            nbrFlat = grid.NbrFlat,
            positions = this.positions,
            invDensities = this.invDensities

        };
        handle = D_job.Schedule(numberOfParticles, 128, handle);


        // 6) Compute pressures from densities
        ComputePressure P0_job = new ComputePressure
        {
            physicsParams = physicsParams,
            eosType = eosType,
            densityClamping = densityClamping,
            pressures = pressures,
            densities = densities
        };
        handle = P0_job.Schedule(numberOfParticles, 128, handle);


        // 7) Compute pressure + viscosity + gravity + XSPH contributions in one pass
        ComputeForcesAndXSPH forcesJob = new ComputeForcesAndXSPH
        {
            integrationParams = intParams,
            phys = physicsParams,
            sph = sphParams,
            pressures = pressures,
            densities = this.densities,
            nbrStart = grid.NbrStart,
            nbrEnd = grid.NbrEnd,
            nbrFlat = grid.NbrFlat,
            positions = this.positions,
            invDensities = this.invDensities,
            forces = this.forces,
            velocities = velocities,
            xsphDelta = xsphDelta,

        };
        handle = forcesJob.Schedule(numberOfParticles, 128, handle);


        // 8) Integrate and handle collisions (symplectic Euler)
        IntegrateWithXSPH integrationJob = new IntegrateWithXSPH
        {
            collisionParams = collisionParams,
            dParams = dampingParams,
            intParams = intParams,
            phys = physicsParams,
            sph = sphParams,
            densities = densities,
            velocities = velocities,
            positions = positions,
            forces = forces,
            rotationMatrix = rotationMatrix,
            rotationMatrixInverse = rotationMatrixInverse,
            xsphDelta = xsphDelta,
            perThreadMaxDispSq = perThreadMaxDispSq,
        };
        handle = integrationJob.Schedule(numberOfParticles, 128, handle);

        if (displacementGridRebuild)
        {
            CheckIfBuildGrid buildJob = new CheckIfBuildGrid
            {
                accumulatedDisplacement = accumulatedDisplacement,
                deltaH = deltaH,
                perThreadMaxDispSq = perThreadMaxDispSq,
                rebuildGrid = rebuildGrid,
            };
            handle = buildJob.Schedule(handle);
        }

        handle.Complete();
    }
    private void BuildGridAndNbr()
    {


        // 2) Build grid & neighbor lists and permute
        handle = grid.BuildGrid(positions, numberOfParticles, handle);

        //permute all arrays so they have same order has particleCount
        PermuteAllArrays permuteJob = new PermuteAllArrays
        {
            perm = grid.ParticlesIndex,
            srcPos = positions,
            dstPos = positions_tmp,
            srcVel = velocities,
            dstVel = velocities_tmp,
        };
        handle = permuteJob.Schedule(numberOfParticles, 128, handle);

        SwapAllArrays swapJob = new SwapAllArrays
        {
            srcPos = positions_tmp,
            srcVel = velocities_tmp,
            dstPos = positions,
            dstVel = velocities
        };

        handle = swapJob.Schedule(numberOfParticles, 128, handle);
        handle = grid.BuildNeighbors(positions, numberOfParticles, h, handle);
        handle.Complete();

    }

    private void Predictive_Corrective_WCSPH(float dt)
    {
        // 1) Update box transform for collision detection
        centerBox = transform.position;
        rotationMatrix = new float3x3(transform.rotation);
        rotationMatrixInverse = math.transpose(rotationMatrix);


        // 2) Backup velocities for viscosity & artificial viscosity (prediction)
        var copyJob = new CopyVelocitiesPositions
        {
            sourceVel = velocities,
            destVel = velocities_saved,
            destPos = positions_prev,
            sourcePos = positions
        };
        handle = copyJob.Schedule(numberOfParticles, 128, handle);


        // 3) Densities
        CalculateDensities D_job = new CalculateDensities
        {
            physicsParams = physicsParams,
            sphParams = sphParams,
            densities = this.densities,
            nbrStart = grid.NbrStart,
            nbrEnd = grid.NbrEnd,
            nbrFlat = grid.NbrFlat,
            positions = this.positions,
            invDensities = this.invDensities

        };
        handle = D_job.Schedule(numberOfParticles, 128, handle);


        // 4) Predictor: viscosity + gravity
        ComputeViscosity visJob = new ComputeViscosity
        {
            sphParams = sphParams,
            physicsParams = physicsParams,
            dt = dt,
            forces = this.forces,
            invDensities = this.invDensities,
            positions = this.positions,
            velocities_out = velocities,
            velocities_in = velocities_saved,
            nbrStart = grid.NbrStart,
            nbrEnd = grid.NbrEnd,
            nbrFlat = grid.NbrFlat,
        };
        handle = visJob.Schedule(numberOfParticles, 128, handle);


        // 5) Pressures
        ComputePressure P0_job = new ComputePressure
        {
            physicsParams = physicsParams,
            densityClamping = this.densityClamping,
            eosType = eosType,
            pressures = pressures,
            densities = densities
        };
        handle = P0_job.Schedule(numberOfParticles, 128, handle);


        // 6) Corrector: pressure forces using old velocity
        ComputePressureForces P1_job = new ComputePressureForces
        {
            physicsParams = physicsParams,
            sphParams = sphParams,
            pressures = pressures,
            invDensities = invDensities,
            positions = positions,
            velocities = velocities_saved,
            nbrEnd = grid.NbrEnd,
            nbrFlat = grid.NbrFlat,
            nbrStart = grid.NbrStart,
            forces = forces
        };
        handle = P1_job.Schedule(numberOfParticles, 128, handle);


        if (useXSph && xsphAppliesToPosition)
        {
            // 7) Integrate 
            IntegrateVelocityOnly IntegrationJob = new IntegrateVelocityOnly
            {
                intParams = intParams,
                cParams = collisionParams,
                dParams = dampingParams,
                phys = physicsParams,
                densities = densities,
                velocities = velocities,
                forces = forces,
                rotationMatrix = rotationMatrix,
                rotationMatrixInverse = rotationMatrixInverse,


            };
            handle = IntegrationJob.Schedule(numberOfParticles, 128, handle);


            XSPH_Accumulate xsphAcc = new XSPH_Accumulate
            {
                intParams = intParams,
                phys = physicsParams,
                sph = sphParams,
                positions = positions,
                velocities = velocities,
                densities = densities,
                nbrStart = grid.NbrStart,
                nbrEnd = grid.NbrEnd,
                nbrFlat = grid.NbrFlat,
                xsphDelta = xsphDelta
            };
            handle = xsphAcc.Schedule(numberOfParticles, 128, handle);


            AdvectPositionsWithXSPH xsphCollisionJob = new AdvectPositionsWithXSPH
            {
                cParams = collisionParams,
                intParams = intParams,
                positions = positions,
                velocities = velocities,
                xsphDelta = xsphDelta,
                rotationMatrix = rotationMatrix,
                rotationMatrixInverse = rotationMatrixInverse,
            };
            handle = xsphCollisionJob.Schedule(numberOfParticles, 128, handle);

            if (displacementGridRebuild)
            {
                CheckIfBuildGrid buildJob = new CheckIfBuildGrid
                {
                    accumulatedDisplacement = accumulatedDisplacement,
                    deltaH = deltaH,
                    perThreadMaxDispSq = perThreadMaxDispSq,
                    rebuildGrid = rebuildGrid,
                };
                handle = buildJob.Schedule(handle);
            }

            handle.Complete();
        }
        else
        {

            // 7) Integrate + collisions
            IntegrateVelocityAndPosition I_job = new IntegrateVelocityAndPosition
            {
                cParams = collisionParams,
                dParams = dampingParams,
                intParams = intParams,
                phys = physicsParams,
                sph = sphParams,
                densities = densities,
                velocities = velocities,
                positions = positions,
                forces = forces,
                rotationMatrix = rotationMatrix,
                rotationMatrixInverse = rotationMatrixInverse,

            };
            handle = I_job.Schedule(numberOfParticles, 128, handle);


            // 8) Optional XSPH smoothing
            if (useXSph)
            {
                XSPH_Accumulate xsphAcc = new XSPH_Accumulate
                {
                    intParams = intParams,
                    phys = physicsParams,
                    sph = sphParams,
                    positions = positions,
                    velocities = velocities,
                    densities = densities,
                    nbrStart = grid.NbrStart,
                    nbrEnd = grid.NbrEnd,
                    nbrFlat = grid.NbrFlat,
                    xsphDelta = xsphDelta
                };
                handle = xsphAcc.Schedule(numberOfParticles, 128, handle);


                ApplyXSPHToVelocity xsphApply = new ApplyXSPHToVelocity
                {
                    xsphDelta = xsphDelta,
                    velocities = velocities
                };
                handle = xsphApply.Schedule(numberOfParticles, 128, handle);


            }
            if (displacementGridRebuild)
            {
                CheckIfBuildGrid buildJob = new CheckIfBuildGrid
                {
                    accumulatedDisplacement = accumulatedDisplacement,
                    deltaH = deltaH,
                    perThreadMaxDispSq = perThreadMaxDispSq,
                    rebuildGrid = rebuildGrid,
                };
                handle = buildJob.Schedule(handle);
            }

            handle.Complete();
        }

    }


    //Draw box using gizmos.
    void OnDrawGizmos()
    {

        if (this.enabled)
        {
            Gizmos.color = Color.red;
            Gizmos.matrix = Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one);
            Gizmos.DrawWireCube(Vector3.zero, boundsSize);
        }

    }


    private void OnDestroy()
    {
        //Disposing native arrays.
        destroyArrays();
    }



    // ------------------------------------------------------------------
    // Helper functions
    // ------------------------------------------------------------------
    private void UpdateParams()
    {
        var rotation = new float3x3(transform.rotation);

        physicsParams = new PhysicsParams
        {
            mass = mass,
            invMass = 1f / mass,

            viscosity = viscosity,

            gamma = gamma,
            soundSpeed = soundSpeed,
            restDensity = restDensity,
            pressureStiff = pressureStiff,

            gravity = new float3(0, -gravity, 0),

            avAlpha = avAlpha,
            avBeta = avBeta,
            avEps = avEps
        };
        //Linear velocity of box
        float3 deltaPos = (float3)transform.position - boxPrevPos;
        float3 velocityBox = deltaPos / simDt;

        //Angular velocity of box
        Quaternion deltaRotBox = transform.rotation * Quaternion.Inverse(qPrevBox);
        deltaRotBox.ToAngleAxis(out float angleDeg, out Vector3 axis);
        float angleRad = angleDeg * Mathf.Deg2Rad;
        Vector3 angularVelocity = axis * (angleRad / simDt);

        boxPrevPos = transform.position;
        qPrevBox = transform.rotation;

        collisionParams = new CollisionParams
        {

            boundsSize = boundsSize,
            boxPosition = transform.position,
            collisionRadius = collisionRadius,
            collisionDamping = collisionDamping,
            angularVelocity = angularVelocity,
            velocity = velocityBox
        };

        dampingParams = new DampingParams
        {
            enabled = densityDamping,
            speedThreshold = speedThreshold,
            densityErrorThreshold = densityErrorThreshold,
            factor = restDampingFactor,
        };

        intParams = new IntegrationParams
        {

            dt = simDt,
            maxVelocity = maxVelocity,
            xsphAppliesToPosition = xsphAppliesToPosition,
            useXSPH = useXSph,
            xsphEps = xsphEps
        };
    }

    //Create render parameter object
    private void CreateRp()
    {
        material.enableInstancing = true;
        var worldCenter = transform.position;
        var extent = boundsSize + (new float3(1, 1, 1) * (4f * h));
        var bb = new Bounds(worldCenter, extent);

        rp = new RenderParams(material)
        {
            layer = gameObject.layer,

            worldBounds = bb
        };
    }


    //Compute parameters and constants at awake.
    private void CalculateParameters()
    {
        //Physics
        avBeta = 2 * avAlpha;
        h = particleSpacing * eta;
        h2 = h * h;
        mass = restDensity * particleSpacing * particleSpacing * particleSpacing;
        collisionRadius = 0.1f * particleSpacing;
        numberOfParticles = particlesPerAxis * particlesPerAxis * particlesPerAxis;


        sphereMesh = Sph.SphereGenerator.GenerateSphereMesh(resolution); //Generate the sphere mesh
        radius = particleSpacing * ResizeFactor;

        grid = new CSR_Grid(numberOfParticles, h);

        sphParams = new SPHParams
        {
            h = h,
            h2 = h2,
            spikyCoeff = 15f / (Mathf.PI * h * h * h * h * h * h),
            poly6Coeff = 315f / (64f * Mathf.PI * h2 * h2 * h2 * h * h * h),
            viscLapCoeff = 45 / (Mathf.PI * h2 * h2 * h2),
            wendlandC2Coeff = 21f / (2f * math.PI * h * h * h),
            numberOfParticles = numberOfParticles
        };

        UpdateParams();
    }


    //Dispose of arrays 
    private void destroyArrays()
    {
        handle.Complete();
        xsphDelta.Dispose();
        xsphDelta_tmp.Dispose();
        positions.Dispose();
        positions_tmp.Dispose();
        velocities_tmp.Dispose();
        velocities.Dispose();
        forces.Dispose();
        pressureForce_tmp.Dispose();
        densities.Dispose();
        densities_tmp.Dispose();
        invDensities.Dispose();
        invDensities_tmp.Dispose();
        pressures.Dispose();
        pressures_tmp.Dispose();
        maxVel.Dispose();
        rebuildGrid.Dispose();
        positions_prev.Dispose();
        velocities_saved.Dispose();
        velocities_old_tmp.Dispose();
        perThreadMaxDispSq.Dispose();
        grid.Dispose();
        accumulatedDisplacement.Dispose();
    }


    //Initialise all arrays 
    private void InitialiseArraysAndPositions()
    {
        xsphDelta = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        xsphDelta_tmp = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        accumulatedDisplacement = new NativeReference<float>(Allocator.Persistent);
        perThreadMaxDispSq = new NativeArray<float>(JobsUtility.MaxJobThreadCount, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        rebuildGrid = new NativeReference<bool>(Allocator.Persistent);
        maxVel = new NativeReference<float>(Allocator.Persistent);
        positions = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        positions_prev = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        positions_tmp = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        velocities = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        velocities_tmp = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        forces = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        pressureForce_tmp = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        velocities_saved = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        velocities_old_tmp = new NativeArray<float3>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        densities = new NativeArray<float>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        densities_tmp = new NativeArray<float>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        invDensities = new NativeArray<float>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        invDensities_tmp = new NativeArray<float>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        pressures = new NativeArray<float>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        pressures_tmp = new NativeArray<float>(numberOfParticles, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        mats = new Matrix4x4[numberOfParticles];



        int numberXAxis = particlesPerAxis;
        int numberYAxis = particlesPerAxis;
        int numberZAxis = particlesPerAxis;

        float3 blockSize = new float3((numberXAxis - 1) * particleSpacing, (numberYAxis - 1) * particleSpacing, (numberZAxis - 1) * particleSpacing);
        float3 start = (float3)transform.position - (0.5f * blockSize);
        start = start + new float3(0, height, 0);

        //Initial positions
        int count = 0;
        for (int i = 0; i < numberXAxis; i++)
        {
            for (int j = 0; j < numberYAxis; j++)
            {
                for (int k = 0; k < numberZAxis; k++)
                {
                    float3 pos = start + (new float3(i, j, k) * particleSpacing);
                    float3 jitter = (jitterStrength > 0f)
                       ? UnityEngine.Random.insideUnitSphere * (jitterStrength * 0.05f * particleSpacing)
                       : float3.zero;

                    positions[count] = pos + jitter;
                    velocities[count] = float3.zero;
                    count++;
                }

            }
        }
    }


    //Scales the mesh for the particles.
    void ScaleMesh(Mesh mesh)
    {
        Vector3[] vertices = mesh.vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i] = Vector3.Scale(vertices[i], new Vector3(radius, radius, radius));
        }
        mesh.vertices = vertices;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }


    public static void CheckCollision(int i, float deltaT, float3x3 rotationMatrix, float3x3 rotationMatrixInverse
    , NativeArray<float3> positions, NativeArray<float3> velocities, float collisionDamping, float3 boundsSize
        , float3 boxPosition, float collisionRadius, float maxSpeed, float3 linearVelB, float3 angularVelB)
    {
        //Calculate local position
        float3 half = boundsSize * 0.5f;
        float3 localPos = math.mul(rotationMatrixInverse, positions[i] - boxPosition);



        //Collrad
        float3 minL = -half + collisionRadius;
        float3 maxL = half - collisionRadius;
        float3 nLocal = 0;

        float3 pNewLocal = localPos;

        // penetration on each axis
        if (pNewLocal.x < minL.x)
        {
            pNewLocal.x = minL.x;
            nLocal += new float3(1, 0, 0);
        }
        else if (pNewLocal.x > maxL.x)
        {
            pNewLocal.x = maxL.x;
            nLocal += new float3(-1, 0, 0);
        }

        if (pNewLocal.y < minL.y)
        {
            pNewLocal.y = minL.y;
            nLocal += new float3(0, 1, 0);
        }
        else if (pNewLocal.y > maxL.y)
        {
            pNewLocal.y = maxL.y;
            nLocal += new float3(0, -1, 0);
        }

        if (pNewLocal.z < minL.z)
        {
            pNewLocal.z = minL.z; nLocal += new float3(0, 0, 1);
        }
        else if (pNewLocal.z > maxL.z)
        {
            pNewLocal.z = maxL.z;
            nLocal += new float3(0, 0, -1);
        }

        if (math.lengthsq(nLocal) > 0)
        {
            nLocal = math.normalize(nLocal);
            float3 pNewWorld = math.mul(rotationMatrix, pNewLocal) + boxPosition;

            //Find box velocity in world coords at contact point
            float3 dCenter = pNewWorld - boxPosition;
            float3 vBox = linearVelB + math.cross(angularVelB, dCenter);
            float3 vRel = velocities[i] - vBox;

            //Normal in world space
            float3 nWorld = math.normalize(math.mul(rotationMatrix, nLocal));

            // collision only if we’re moving into the wall
            float vRelN = math.dot(vRel, nWorld);
            if (vRelN < 0)
            {
                float friction = 0f;

                float3 vN = vRelN * nWorld;
                float3 vT = vRel - vN;

                vRel = (-(1 - collisionDamping) * vN) + ((1 - friction) * vT);
            }

            velocities[i] = vRel + vBox;
            positions[i] = pNewWorld;
        }

    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ClampVelocity(int i, NativeArray<float3> velocities, float maxVelocity)
    {
        float3 v = velocities[i];
        float speedSquared = math.lengthsq(v);

        float maxSpeedSquared = maxVelocity * maxVelocity;
        if (speedSquared <= maxSpeedSquared)
            return;

        // Scale down to max velocity
        float speed = math.sqrt(speedSquared);
        velocities[i] = v * (maxVelocity / speed);
    }


    public static void densityDampingMeth(int i, NativeArray<float> densities, NativeArray<float3> velocities, float initialDensity
        , float restDampingThresholdSpeed, float densityErrorThreshold, float restDampingFactor)
    {
        float speed = math.length(velocities[i]);
        float densityError = math.abs(densities[i] - initialDensity) / initialDensity;

        if (speed < restDampingThresholdSpeed && densityError < densityErrorThreshold)
        {
            velocities[i] *= restDampingFactor;
        }
    }



    // ------------------------------------------------------------------
    // Forces jobs
    // ------------------------------------------------------------------



    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct CalculateDensities : IJobParallelFor
    {
        public SPHParams sphParams;
        public PhysicsParams physicsParams;
        public NativeArray<float> densities;


        [WriteOnly] public NativeArray<float> invDensities;
        [ReadOnly] public NativeArray<int> nbrStart, nbrEnd, nbrFlat;
        [ReadOnly] public NativeArray<float3> positions;
        public void Execute(int i)
        {
            // Self contribution
            float density = physicsParams.mass * Kernels_SPH.WendlandC2(sphParams.h, sphParams.h2, 0, sphParams.wendlandC2Coeff);
            // Neighbors
            int start = nbrStart[i];
            int end = nbrEnd[i];
            float3 p_i = positions[i];

            // Overall density computation across all particles is O(N * K).
            for (int k = start; k < end; k++)
            {

                int j = nbrFlat[k];
                float3 r_ij = p_i - positions[j];
                float d2 = math.lengthsq(r_ij);
                if (d2 >= sphParams.h2)
                    continue;

                density += SPHForces.wendlandC2Density(sphParams, physicsParams.mass, d2);
            }

            densities[i] = density;
            invDensities[i] = 1f / math.max(density, 1e-6f);
        }


    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct ComputeViscosity : IJobParallelFor
    {
        public SPHParams sphParams;
        public PhysicsParams physicsParams;

        public float dt;

        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float> invDensities;
        public NativeArray<float3> forces;
        [ReadOnly] public NativeArray<int> nbrStart, nbrEnd, nbrFlat;
        [ReadOnly] public NativeArray<float3> velocities_in;
        [WriteOnly] public NativeArray<float3> velocities_out;
        public void Execute(int i)
        {

            float3 viscosityForce = float3.zero;
            int start = nbrStart[i];
            int end = nbrEnd[i];

            float3 pos_i = positions[i];
            float rho_i = 1f / invDensities[i];
            float3 vel_i = velocities_in[i];

            for (int k = start; k < end; k++)
            {
                int j = nbrFlat[k];
                float3 r_ij = pos_i - positions[j];
                float d2 = math.lengthsq(r_ij);
                if (d2 >= sphParams.h2)
                    continue;

                float3 v_ji = velocities_in[j] - vel_i;
                float rho_j = 1f / invDensities[j];

                viscosityForce += SPHForces.SymetricMonaghanViscosity(sphParams, physicsParams, d2, v_ji, rho_j, rho_i);
            }

            //Predicted velocity
            velocities_out[i] = vel_i + (dt * (physicsParams.gravity + (physicsParams.invMass * viscosityForce)));

        }

    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct ComputePressure : IJobParallelFor
    {
        public PhysicsParams physicsParams;
        public EOS eosType;
        public bool densityClamping;
        [WriteOnly] public NativeArray<float> pressures;
        [ReadOnly] public NativeArray<float> densities;
        public void Execute(int i)
        {
            float density_i = densities[i];
            float pressure = 0;

            // Clamping density if option is enabled.
            // Helps stability for Tait EOS; for the linear EOS it's usually better disabled.

            if (densityClamping)
            {
                density_i = math.clamp(density_i, minRatio * physicsParams.restDensity,
                    maxRatio * physicsParams.restDensity);

            }

            // Equation of state
            switch (eosType)
            {
                case EOS.Tait:
                    pressure = PressureSolvers.TaitPressure(density_i, physicsParams.restDensity
                        , physicsParams.soundSpeed, physicsParams.gamma);
                    break;
                case EOS.Linear:
                    pressure = PressureSolvers.LinearEOS(density_i, physicsParams.restDensity
                        , physicsParams.pressureStiff);
                    break;
            }

            pressures[i] = pressure;
        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct ComputePressureForces : IJobParallelFor
    {
        // Constant params for all particles
        public SPHParams sphParams;
        public PhysicsParams physicsParams;
        // Per-particle data
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<float3> velocities;
        [ReadOnly] public NativeArray<float> pressures;
        [ReadOnly] public NativeArray<float> invDensities;
        [ReadOnly] public NativeArray<int> nbrStart, nbrEnd, nbrFlat;
        public NativeArray<float3> forces;

        public void Execute(int i)
        {
            float3 pos_i = positions[i];
            float3 vel_i = velocities[i];
            float p_i = pressures[i];
            float invRho_i = invDensities[i];

            float3 force = float3.zero;

            int start = nbrStart[i];
            int end = nbrEnd[i];

            for (int idx = start; idx < end; idx++)
            {
                int j = nbrFlat[idx];

                float3 r_ij = pos_i - positions[j];
                float d2 = math.lengthsq(r_ij);

                if (d2 < 1e-12f || d2 >= sphParams.h2)
                    continue;

                float3 v_ij = vel_i - velocities[j];



                float vr = math.dot(v_ij, r_ij);
                float p_j = pressures[j];
                float invRho_j = invDensities[j];



                force += SPHForces.PressureForceWithArtificialViscosity(sphParams, physicsParams, r_ij,
                    p_i, invRho_i, p_j, invRho_j, vr, d2);
            }

            forces[i] = force;
        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct ComputeForcesAndXSPH : IJobParallelFor
    {
        public SPHParams sph;
        public PhysicsParams phys;
        public IntegrationParams integrationParams;

        //PressureForces
        [ReadOnly] public NativeArray<float> pressures, invDensities;
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<int> nbrStart, nbrEnd, nbrFlat;
        public NativeArray<float3> forces;

        //ViscosityForces
        [ReadOnly] public NativeArray<float3> velocities;

        [ReadOnly] public NativeArray<float> densities;
        [WriteOnly] public NativeArray<float3> xsphDelta;

        public void Execute(int i)
        {
            float3 pos_i = positions[i];
            float3 vel_i = velocities[i];
            float p_i = pressures[i];
            float invRho_i = invDensities[i];
            float rho_i = math.max(densities[i], 1e-6f);

            // Accumulators for all three computations
            float3 pressureForce = float3.zero;
            float3 viscosityForce = float3.zero;
            float3 xsphCorr = float3.zero;

            int s = nbrStart[i], e = nbrEnd[i];

            for (int idx = s; idx < e; idx++)
            {
                int j = nbrFlat[idx];


                float3 r_ij = pos_i - positions[j];
                float d2 = math.lengthsq(r_ij);


                if (d2 >= sph.h2 || d2 <= 1e-12f)
                    continue;


                float invRho_j = invDensities[j];
                float p_j = pressures[j];
                float3 v_ij = vel_i - velocities[j];
                float vr = math.dot(v_ij, r_ij);


                //Compute pressure and viscosity artificial
                pressureForce += SPHForces.PressureForceWithArtificialViscosity(sph, phys,
                    r_ij, p_i, invRho_i, p_j, invRho_j, vr, d2);


                // Physical viscosity
                float3 v_ji = -v_ij;
                viscosityForce += SPHForces.SymetricMonaghanViscosity(sph, phys,
                    d2, v_ji, densities[j], rho_i);


                //  XSPH
                float rho_j = math.max(densities[j], 1e-6f);
                xsphCorr += SPHForces.xsphCorrection(sph, phys, d2, v_ji, rho_i, rho_j);
            }

            // Accumulate all forces
            forces[i] = pressureForce + (phys.viscosity * viscosityForce)
                + (phys.mass * phys.gravity);
            xsphDelta[i] = integrationParams.xsphEps * xsphCorr;
        }

    }


    // ------------------------------------------------------------------
    // Integration jobs
    // ------------------------------------------------------------------


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct IntegrateVelocityAndPosition : IJobParallelFor
    {
        public SPHParams sph;
        public PhysicsParams phys;
        public CollisionParams cParams;
        public IntegrationParams intParams;
        public DampingParams dParams;

        public NativeArray<float3> velocities, positions;
        public float3x3 rotationMatrix, rotationMatrixInverse;
        [ReadOnly] public NativeArray<float3> forces;
        [ReadOnly] public NativeArray<float> densities;
        public void Execute(int i)
        {

            float3 oldPos = positions[i];
            // integrate velocities first
            velocities[i] += intParams.dt * phys.invMass * forces[i];

            //Velocity clamping
            ClampVelocity(i, velocities, intParams.maxVelocity);
            //Damping close to rest
            if (dParams.enabled)
            {
                densityDampingMeth(i, densities, velocities, phys.restDensity,
                    dParams.speedThreshold, dParams.densityErrorThreshold, dParams.factor);
            }

            // then positions
            positions[i] += intParams.dt * velocities[i];

            CheckCollision(i, intParams.dt, rotationMatrix, rotationMatrixInverse,
                positions, velocities, cParams.collisionDamping, cParams.boundsSize
                , cParams.boxPosition, cParams.collisionRadius, intParams.maxVelocity, cParams.velocity, cParams.angularVelocity);
        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct IntegrateVelocityOnly : IJobParallelFor
    {
        public CollisionParams cParams;
        public PhysicsParams phys;
        public IntegrationParams intParams;
        public DampingParams dParams;

        public NativeArray<float3> velocities;
        public float3x3 rotationMatrix, rotationMatrixInverse;
        [ReadOnly] public NativeArray<float3> forces;
        [ReadOnly] public NativeArray<float> densities;
        public void Execute(int i)
        {

            // integrate velocities first
            velocities[i] += intParams.dt * phys.invMass * forces[i];

            //Velocity clamping
            ClampVelocity(i, velocities, intParams.maxVelocity);

            //Damping close to rest
            if (dParams.enabled)
            {
                densityDampingMeth(i, densities, velocities, phys.restDensity,
                    dParams.speedThreshold, dParams.densityErrorThreshold, dParams.factor);
            }


        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct IntegrateWithXSPH : IJobParallelFor
    {
        public SPHParams sph;
        public PhysicsParams phys;
        public IntegrationParams intParams;
        public DampingParams dParams;
        public CollisionParams collisionParams;

        public NativeArray<float3> velocities, positions;
        public float3x3 rotationMatrix, rotationMatrixInverse;
        [ReadOnly] public NativeArray<float3> forces;
        [ReadOnly] public NativeArray<float3> xsphDelta;
        [ReadOnly] public NativeArray<float> densities;

        [NativeDisableContainerSafetyRestriction]
        public NativeArray<float> perThreadMaxDispSq;

        [NativeSetThreadIndex]
        int threadIndex;
        public void Execute(int i)
        {
            float3 oldPosition = positions[i];
            velocities[i] += intParams.dt * phys.invMass * forces[i];

            // We use XSPH has an artificial viscosity term (disipative force)
            if (intParams.useXSPH && !intParams.xsphAppliesToPosition)
                velocities[i] += xsphDelta[i];

            ClampVelocity(i, velocities, intParams.maxVelocity);


            // Damps velocity if particle is close to rest density
            if (dParams.enabled)
            {
                densityDampingMeth(i, densities, velocities, phys.restDensity,
                    dParams.speedThreshold, dParams.densityErrorThreshold, dParams.factor);
            }


            // then positions
            positions[i] += intParams.dt * velocities[i];

            //We use XSPH to smooth fluid witouth affecting momentum.
            if (intParams.xsphAppliesToPosition && intParams.useXSPH)
                positions[i] += intParams.dt * xsphDelta[i];

            CheckCollision(i, intParams.dt, rotationMatrix, rotationMatrixInverse, positions, velocities
                , collisionParams.collisionDamping, collisionParams.boundsSize
               , collisionParams.boxPosition, collisionParams.collisionRadius, intParams.maxVelocity, collisionParams.velocity, collisionParams.angularVelocity);

            float displacementSq = math.lengthsq(oldPosition - positions[i]);
            if (perThreadMaxDispSq[threadIndex] < displacementSq)
                perThreadMaxDispSq[threadIndex] = displacementSq;
        }


    }


    //Monaghan
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct XSPH_Accumulate : IJobParallelFor
    {
        public SPHParams sph;
        public PhysicsParams phys;
        public IntegrationParams intParams;

        [ReadOnly] public NativeArray<float3> positions, velocities;
        [ReadOnly] public NativeArray<float> densities;
        [ReadOnly] public NativeArray<int> nbrStart, nbrEnd, nbrFlat;
        [WriteOnly] public NativeArray<float3> xsphDelta;



        public void Execute(int i)
        {
            float3 corr = float3.zero;
            float rho_i = math.max(densities[i], 1e-6f);

            int s = nbrStart[i], e = nbrEnd[i];
            float3 pi = positions[i];
            float3 vi = velocities[i];

            for (int idx = s; idx < e; idx++)
            {
                int j = nbrFlat[idx];
                float3 r_ij = pi - positions[j];
                float d2 = math.lengthsq(r_ij);
                if (d2 >= sph.h2)
                    continue;

                float rho_j = math.max(densities[j], 1e-6f);
                float3 v_ji = velocities[j] - vi;

                corr += SPHForces.xsphCorrection(sph, phys, d2, v_ji, rho_i, rho_j);
            }

            xsphDelta[i] = intParams.xsphEps * corr;
        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct ApplyXSPHToVelocity : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> xsphDelta;
        public NativeArray<float3> velocities;

        public void Execute(int i)
        {
            velocities[i] += xsphDelta[i];
        }
    }



    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct AdvectPositionsWithXSPH : IJobParallelFor
    {
        public CollisionParams cParams;
        public IntegrationParams intParams;

        [ReadOnly] public NativeArray<float3> xsphDelta;

        public NativeArray<float3> positions, velocities;
        public float3x3 rotationMatrix, rotationMatrixInverse;


        public void Execute(int i)
        {
            positions[i] += intParams.dt * (velocities[i] + xsphDelta[i]);

            CheckCollision(i, intParams.dt, rotationMatrix, rotationMatrixInverse, positions, velocities
                , cParams.collisionDamping, cParams.boundsSize
                , cParams.boxPosition, cParams.collisionRadius, intParams.maxVelocity, cParams.velocity, cParams.angularVelocity);

        }
    }


    // ------------------------------------------------------------------
    // Helper jobs
    // ------------------------------------------------------------------

    [BurstCompile(FloatMode = FloatMode.Fast)]
    public struct PermuteAllArrays : IJobParallelFor
    {
        [ReadOnly] public NativeArray<int> perm;

        // Sources
        [ReadOnly] public NativeArray<float3> srcPos, srcVel;

        // Destinations
        [WriteOnly] public NativeArray<float3> dstPos, dstVel;

        public void Execute(int i)
        {
            int src = perm[i];
            dstPos[i] = srcPos[src];
            dstVel[i] = srcVel[src];

        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct CopyVelocitiesPositions : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> sourceVel, sourcePos;
        [WriteOnly] public NativeArray<float3> destVel, destPos;

        public void Execute(int i)
        {
            destVel[i] = sourceVel[i];
            destPos[i] = sourcePos[i];
        }
    }
    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct MaxVelocity : IJob
    {
        [ReadOnly] public NativeArray<float3> velocities;
        [WriteOnly] public NativeReference<float> maxVel;
        public void Execute()
        {
            float maxVelocity = 0;
            for (int i = 0; i < velocities.Length; i++)
            {
                float velMag = math.length(velocities[i]);
                if (velMag > maxVelocity)
                {
                    maxVelocity = velMag;
                }
            }
            maxVel.Value = maxVelocity;
        }
    }

    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    struct ClearForces : IJobParallelFor
    {
        public NativeArray<float3> a;
        public void Execute(int i)
        {
            a[i] = 0f;

        }
    }
    [BurstCompile(FloatMode = FloatMode.Fast)]
    public struct SwapAllArrays : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float3> srcPos, srcVel;
        [WriteOnly] public NativeArray<float3> dstPos, dstVel;
        public void Execute(int i)
        {
            dstPos[i] = srcPos[i];
            dstVel[i] = srcVel[i];
        }
    }
    [BurstCompile(FloatMode = FloatMode.Fast, OptimizeFor = OptimizeFor.Performance)]
    public struct CheckIfBuildGrid : IJob
    {
        public float deltaH;
        public NativeArray<float> perThreadMaxDispSq;
        [WriteOnly] public NativeReference<bool> rebuildGrid;
        public NativeReference<float> accumulatedDisplacement;
        public void Execute()
        {
            float threshold = deltaH / 3f;
            float maxDispSq = 0f;

            for (int i = 0; i < perThreadMaxDispSq.Length; i++)
            {
                maxDispSq = math.max(maxDispSq, perThreadMaxDispSq[i]);
                perThreadMaxDispSq[i] = 0f;
            }

            accumulatedDisplacement.Value += math.sqrt(maxDispSq);
            rebuildGrid.Value = accumulatedDisplacement.Value > threshold;
        }
    }
}



