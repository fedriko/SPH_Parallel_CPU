using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

/// <summary>
/// Uniform grid + CSR neighbor structure for SPH.
/// </summary>
public class CSR_Grid : System.IDisposable
{
    private const int EstimatedMaxNeighborsPerParticle = 64;
    private readonly Allocator allocator;

    //World-space size of a grid cell.
    private float cellSize;
    private float invCellSize;



    // Maximum number of particles 
    private int capacity;


    //Integer cell coordinates for each particle
    private NativeArray<int3> cellCoords;

    //Linear cell ID for each particle (before sorting)
    private NativeArray<int> cellKey;

    // <summary>Particle indices sorted by cellKey
    private NativeArray<int> particlesIndex;

    //Sorted keys matching particlesIndex
    private NativeArray<int> sortedKey;

    //Temp arrays for radix sort
    private NativeArray<int> sortedValues;
    private NativeArray<int> countP;
    private NativeArray<int> prefix;
    private NativeArray<int> tmpKeys;
    private NativeArray<int> tmpVals;

    // Grid dimensions as NativeReferences 
    private NativeReference<int> sizeX, sizeY, sizeZ;
    private NativeReference<int> totalCellsRef;

    //List of all unique cell keys that are occupied
    private NativeArray<int> uniqueCellKeys;

    //Maps a linear cell key to its compact cell index (or -1 if empty)
    private NativeArray<int> cellIdToCompactIndex;

    // Start & end in particlesIndex
    private NativeArray<int> cellStart;
    private NativeArray<int> cellEnd;

    //Number of unique cells
    private NativeReference<int> uniqueCells;

    //Min-max cell coordinates over all particles.
    private NativeReference<int> minCx, minCy, minCz;
    private NativeReference<int> maxCx, maxCy, maxCz;

    // Neighbor data 

    // Start index in nbrFlat
    private NativeArray<int> nbrStart;

    //End index in nbrFlat (exclusive)
    private NativeArray<int> nbrEnd;

    private NativeArray<int> nbrFlat;

    // Count per particle 
    private NativeArray<int> nbrCount;

    // Offsets of neighbor cells to visit (e.g. 27 offsets (3^3 in 3D))
    private NativeArray<int3> neighborOffsets;

    // Job dependency 

    public JobHandle LastHandle { get; private set; }

    // Public accessors

    public NativeArray<int> NbrStart => nbrStart;
    public NativeArray<int> NbrEnd => nbrEnd;
    public NativeArray<int> NbrFlat => nbrFlat;
    public NativeArray<int> ParticlesIndex => particlesIndex;
    public int GridSizeX => sizeX.Value;
    public int GridSizeY => sizeY.Value;
    public int GridSizeZ => sizeZ.Value;
    public float CellSize => cellSize;
    public float InvCellSize => invCellSize;


    // --------------------------------------------------------------------
    // Construction / initialization
    // --------------------------------------------------------------------

    /// <summary>
    /// Create a SphGrid with an initial capacity and a chosen cell size.
    /// Capacity will grow automatically as needed, but starting near your
    /// expected particle count reduces reallocations.
    /// </summary>
    public CSR_Grid(int initialCapacity, float initialCellSize, Allocator allocator = Allocator.Persistent)
    {
        this.allocator = allocator;
        this.cellSize = initialCellSize;
        this.invCellSize = 1f / initialCellSize;
        this.capacity = math.max(1, initialCapacity);

        AllocateCoreArrays(this.capacity);
        AllocateRefs();
        InitializeNeighborOffsets();
    }

    // Allocate all arrays 
    private void AllocateCoreArrays(int newCapacity)
    {
        // Dispose old arrays if they were allocated
        DisposeCoreArrays();

        // Particle-dependent arrays
        cellCoords = new NativeArray<int3>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        cellKey = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        particlesIndex = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        sortedKey = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        sortedValues = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        countP = new NativeArray<int>(256, allocator, NativeArrayOptions.ClearMemory);
        prefix = new NativeArray<int>(256, allocator, NativeArrayOptions.ClearMemory);
        tmpKeys = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        tmpVals = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        uniqueCellKeys = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);

        nbrStart = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        nbrEnd = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        nbrCount = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.ClearMemory);
        // nbrFlat will be sized after we know total neighbors (CSR build)

        // Cell-dependent arrays: we size these later once we know grid extents
        cellStart = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        cellEnd = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);
        cellIdToCompactIndex = new NativeArray<int>(newCapacity, allocator, NativeArrayOptions.UninitializedMemory);

    }

    private void AllocateRefs()
    {
        // References
        uniqueCells = new NativeReference<int>(allocator);
        minCx = new NativeReference<int>(allocator);
        minCy = new NativeReference<int>(allocator);
        minCz = new NativeReference<int>(allocator);
        maxCx = new NativeReference<int>(allocator);
        maxCy = new NativeReference<int>(allocator);
        maxCz = new NativeReference<int>(allocator);

        // Dimensions
        sizeX = new NativeReference<int>(allocator);
        sizeY = new NativeReference<int>(allocator);
        sizeZ = new NativeReference<int>(allocator);
        totalCellsRef = new NativeReference<int>(allocator);
    }

    // Initialize neighbor cell offsets (27 offsets for 3x3x3 region)
    private void InitializeNeighborOffsets()
    {
        // 3x3x3 stencil around a cell: (-1..1)^3 = 27 offsets
        neighborOffsets = new NativeArray<int3>(27, allocator, NativeArrayOptions.UninitializedMemory);
        int idx = 0;
        for (int z = -1; z <= 1; z++)
            for (int y = -1; y <= 1; y++)
                for (int x = -1; x <= 1; x++)
                    neighborOffsets[idx++] = new int3(x, y, z);
    }




    private void EnsureCapacity(ref NativeArray<int> arr, int required)
    {
        if (!arr.IsCreated)
        {
            arr = new NativeArray<int>(required, allocator, NativeArrayOptions.UninitializedMemory);
            return;
        }

        if (required > arr.Length)
        {
            arr.Dispose();
            arr = new NativeArray<int>(required, allocator, NativeArrayOptions.UninitializedMemory);
        }
    }

    // Build the uniform grid and cell indexing used for neighbor search.
    //
    //    cellCoords[i]           integer cell coordinates for particle i.

    //    cellKey[i]              linearized cell ID for particle i (before sorting).

    //    particlesIndex[]        particle indices sorted by their cellKey.

    //    cellStart[c], cellEnd[c] 
    //                             
    //   - cellIdToCompactIndex[] – maps a linear cell ID to its compact cell index (or -1 if empty).



    
    public JobHandle BuildGrid(NativeArray<float3> positions, int particleCount, JobHandle dependency = default)
    {

        // Resize based on particle count 
        int estimatedCells = Mathf.NextPowerOfTwo(Mathf.Max(64, particleCount * 3));
        EnsureCapacity(ref cellIdToCompactIndex, estimatedCells);
        EnsureCapacity(ref cellStart, estimatedCells);
        EnsureCapacity(ref cellEnd, estimatedCells);
        EnsureCapacity(ref uniqueCellKeys, estimatedCells);

        minCx.Value = int.MaxValue; minCy.Value = int.MaxValue; minCz.Value = int.MaxValue;
        maxCx.Value = int.MinValue; maxCy.Value = int.MinValue; maxCz.Value = int.MinValue;

        //Find the grid size (min and max values)
        FindGridSize gridSize = new FindGridSize
        {
            minCx = minCx,
            minCy = minCy,
            minCz = minCz,
            maxCx = maxCx,
            maxCy = maxCy,
            maxCz = maxCz,
            particleCount = particleCount,
            invCellSize = this.invCellSize,
            positions = positions,
            cellCoords = this.cellCoords
        };
        dependency = gridSize.Schedule(dependency);

        //We need dimension for next task
       // dependency.Complete();

        // 2. Compute dimensions (NO Complete() needed!)
        var setupDimsJob = new SetupGridDimensions
        {
            minCx = minCx,
            minCy = minCy,
            minCz = minCz,
            maxCx = maxCx,
            maxCy = maxCy,
            maxCz = maxCz,
            sizeX = sizeX,
            sizeY = sizeY,
            sizeZ = sizeZ,
            totalCells = totalCellsRef
        };

         dependency = setupDimsJob.Schedule(dependency);
        //Find linear index for cell and construct cellCoords and particlesIndices
        ComputeLinearCellId computeCellId = new ComputeLinearCellId
        {
            minCx = minCx,
            minCy = minCy,
            minCz = minCz,
            sizeX = sizeX,
            sizeY = sizeY,
            cellCoords = this.cellCoords,
            particlesIndices = this.particlesIndex,
            cellKey = this.cellKey,
        };
        dependency = computeCellId.Schedule(particleCount, 128, dependency);


        //Sort cell keys and particlesIndex by cell order using radix sort
        var sortJob = new RadixSort_32bit_signed
        {
            particleCount = particleCount,
            keys = cellKey,
            vals = particlesIndex,
            tmpKeys = sortedKey,
            tmpVals = sortedValues,
            countP = countP,
            prefix = prefix
        };
        dependency = sortJob.Schedule(dependency);


        //Produce contiguous [cellStart, cellEnd) ranges used by CSR neighbor construction.
        BuildFlatGrid buildGrid = new BuildFlatGrid
        {
            particleCount = particleCount,
            UniqueCells = uniqueCells,
            totalCells = totalCellsRef,
            cellStart = this.cellStart,
            cellEnd = this.cellEnd,
            cellKey = this.cellKey,
            uniqueCellKeys = this.uniqueCellKeys,
            cellIdToCompactIndex = this.cellIdToCompactIndex
        };
        dependency = buildGrid.Schedule(dependency);

        LastHandle = dependency;
        return LastHandle;
    }


    // Build the CSR neighbor structure.
    //
    // This creates three arrays:
    //  nbrStart[i] – start index of particle i in nbrFlat
    //  nbrEnd[i]   – end index of particle i in nbrFlat
    //  nbrFlat[]   – flat array containing all neighbor particle indices stored in line
    public JobHandle BuildNeighbors(NativeArray<float3> positions, int particleCount, float h, JobHandle dependency = default)
    {
        float h2 = h * h;
   

        //1) Count neighbors for each particle.
        FindNeighSize job = new FindNeighSize
        {
            h2 = h2,
            minCx = this.minCx,
            minCy = this.minCy,
            minCz = this.minCz,
            positions = positions,
            invCellSize = this.invCellSize,
            neighborOffsets = this.neighborOffsets,
            sizeX = sizeX,
            sizeY = sizeY,
            sizeZ = sizeZ,
            cellIdToRun = this.cellIdToCompactIndex,
            cellStart = this.cellStart,
            cellEnd = this.cellEnd,
            nbrCount = this.nbrCount
        };
        dependency = job.Schedule(particleCount, 128, dependency);


        //2) Prefix - sum the counts to build nbrStart and nbrEnd.
        constructNbrStartAndNbrEnd job2 = new constructNbrStartAndNbrEnd
        {
            particleCount = particleCount,
            nbrStart = this.nbrStart,
            nbrEnd = this.nbrEnd,
            nbrCount = this.nbrCount
        };

        dependency = job2.Schedule(dependency);
        dependency.Complete();

        //3) Allocate or grow nbrFlat to hold all neighbor entries.
         //total number of neighbor entries = end of last particle
        int required = nbrEnd[particleCount - 1];
        EnsureCapacity(ref nbrFlat, required);


        //4) Fill nbrFlat with actual neighbor indices.
        ConstructNbrFlat job3 = new ConstructNbrFlat
        {
            h2 = h2,
            minCx = this.minCx,
            minCy = this.minCy,
            minCz = this.minCz,
            positions = positions,
            invCellSize = this.invCellSize,
            neighborOffsets = this.neighborOffsets,
            sizeX = sizeX,
            sizeY = sizeY,
            sizeZ = sizeZ,
            cellIdToRun = this.cellIdToCompactIndex,
            cellStart = this.cellStart,
            cellEnd = this.cellEnd,
            nbrFlat = this.nbrFlat,
            nbrStart = this.nbrStart,
            nbrEnd = this.nbrEnd
        };
        dependency = job3.Schedule(particleCount, 64, dependency);
        //dependency.Complete();

        LastHandle = dependency;
        return LastHandle;
    }

  
   
    public void Complete()
    {
        LastHandle.Complete();
    }

    //Converts world coordinates to cell coordinates 
    static int3 WorldToCell(float3 p, float invSize)
    {

        return (int3)math.floor(p * invSize);
    }



    public void Dispose()
    {
        LastHandle.Complete();
        DisposeCoreArrays();

        if (neighborOffsets.IsCreated) neighborOffsets.Dispose();
    }

    private void DisposeCoreArrays()
    {
        if (cellCoords.IsCreated) cellCoords.Dispose();
        if (cellKey.IsCreated) cellKey.Dispose();
        if (particlesIndex.IsCreated) particlesIndex.Dispose();
        if (sortedKey.IsCreated) sortedKey.Dispose();
        if (sortedValues.IsCreated) sortedValues.Dispose();
        if (countP.IsCreated) countP.Dispose();
        if (prefix.IsCreated) prefix.Dispose();
        if (tmpKeys.IsCreated) tmpKeys.Dispose();
        if (tmpVals.IsCreated) tmpVals.Dispose();
        if (uniqueCellKeys.IsCreated) uniqueCellKeys.Dispose();
        if (cellIdToCompactIndex.IsCreated) cellIdToCompactIndex.Dispose();
        if (cellStart.IsCreated) cellStart.Dispose();
        if (cellEnd.IsCreated) cellEnd.Dispose();
        if (nbrStart.IsCreated) nbrStart.Dispose();
        if (nbrEnd.IsCreated) nbrEnd.Dispose();
        if (nbrFlat.IsCreated) nbrFlat.Dispose();
        if (nbrCount.IsCreated) nbrCount.Dispose();

        if (uniqueCells.IsCreated) uniqueCells.Dispose();
        if (minCx.IsCreated) minCx.Dispose();
        if (minCy.IsCreated) minCy.Dispose();
        if (minCz.IsCreated) minCz.Dispose();
        if (maxCx.IsCreated) maxCx.Dispose();
        if (maxCy.IsCreated) maxCy.Dispose();
        if (maxCz.IsCreated) maxCz.Dispose();

        if (sizeX.IsCreated) sizeX.Dispose();
        if (sizeY.IsCreated) sizeY.Dispose();
        if (sizeZ.IsCreated) sizeZ.Dispose();
        if (totalCellsRef.IsCreated) totalCellsRef.Dispose();
    }

    // ------------------------------------------------------------------
    // Jobs
    // ------------------------------------------------------------------
    [BurstCompile]
    public struct SetupGridDimensions : IJob
    {
        // Input: min/max cell coordinates from FindGridSize
        [ReadOnly] public NativeReference<int> minCx, minCy, minCz;
        [ReadOnly] public NativeReference<int> maxCx, maxCy, maxCz;

        // Output: computed grid dimensions
        public NativeReference<int> sizeX, sizeY, sizeZ;
        public NativeReference<int> totalCells;

        public void Execute()
        {
            sizeX.Value = maxCx.Value - minCx.Value + 1;
            sizeY.Value = maxCy.Value - minCy.Value + 1;
            sizeZ.Value = maxCz.Value - minCz.Value + 1;
            totalCells.Value = sizeX.Value * sizeY.Value * sizeZ.Value;
        }
    }
    //We find the dimensions of the grid for the current configuration of the fluid. 
    [BurstCompile]
    public struct FindGridSize : IJob
    {
        [WriteOnly] public NativeReference<int> minCx, minCy, minCz;
        [WriteOnly] public NativeReference<int> maxCx, maxCy, maxCz;
        public int particleCount;
        public float invCellSize;
        [ReadOnly] public NativeArray<float3> positions;
        [WriteOnly] public NativeArray<int3> cellCoords;

        public void Execute()
        {
            int lminX = int.MaxValue, lminY = int.MaxValue, lminZ = int.MaxValue;
            int lmaxX = int.MinValue, lmaxY = int.MinValue, lmaxZ = int.MinValue;

            for (int i = 0; i < particleCount; i++)
            {
                int3 c = WorldToCell(positions[i], invCellSize);
                cellCoords[i] = c;
                if (c.x < lminX)
                    lminX = c.x;

                if (c.y < lminY)
                    lminY = c.y;

                if (c.z < lminZ)
                    lminZ = c.z;

                if (c.x > lmaxX)
                    lmaxX = c.x;

                if (c.y > lmaxY)
                    lmaxY = c.y;

                if (c.z > lmaxZ)
                    lmaxZ = c.z;
            }
            minCx.Value = lminX; minCy.Value = lminY; minCz.Value = lminZ;
            maxCx.Value = lmaxX; maxCy.Value = lmaxY; maxCz.Value = lmaxZ;
        }

    }
    public void setCellSize(float cellSize)
    {
        this.cellSize=cellSize;
        this.invCellSize = 1 / cellSize;
    }
    //Helper job to construct the Grid
    //We creater a linear index from our cell index. 
    //(We do this in parallel)
    [BurstCompile]
    public struct ComputeLinearCellId : IJobParallelFor
    {
        [ReadOnly] public NativeReference<int> minCx, minCy, minCz;
        [ReadOnly] public NativeReference<int> sizeX, sizeY;
        [ReadOnly] public NativeArray<int3> cellCoords;
        [WriteOnly] public NativeArray<int> particlesIndices, cellKey;
        public void Execute(int i)
        {


            int3 c = cellCoords[i];

            int linearX = c.x - minCx.Value;
            int linearY = c.y - minCy.Value;
            int linearZ = c.z - minCz.Value;

            // Linear index for cells 
            int id = linearX + sizeX.Value * (linearY + sizeY.Value * linearZ);
            particlesIndices[i] = i;
            cellKey[i] = id;


        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct RadixSort_32bit_signed : IJob
    {
        public int particleCount;
        public NativeArray<int> keys;
        public NativeArray<int> vals;
        public NativeArray<int> tmpKeys;
        public NativeArray<int> tmpVals;
        public NativeArray<int> countP;
        public NativeArray<int> prefix;

        public void Execute()
        {
            int n = particleCount;

            for (int pass = 0; pass < 4; pass++)
            {
                for (int i = 0; i < 256; i++) 
                    countP[i] = 0;

                for (int i = 0; i < n; i++)
                {

                    int shift = pass * 8;
                    //Logic shift left and keep only the first 8 bits
                    int b = (keys[i] >> shift) & 0xFF;

                    // Last pass (most significant byte): fix signed ordering by inversing sign byte. (use XOR)
                    if (pass == 3)
                        b ^= 0x80;

                    countP[b]++;
                }

                int running = 0;
                for (int i = 0; i < 256; i++)
                {
                    prefix[i] = running;
                    running += countP[i];
                }

                for (int i = 0; i < n; i++)
                {
                    int k = keys[i];

                    int shift = pass * 8;
                    int b = (k >> shift) & 0xFF;

                    int pos = prefix[b];
                    prefix[b]++;

                    tmpKeys[pos] = k;
                    tmpVals[pos] = vals[i];
                }

                (keys, tmpKeys) = (tmpKeys, keys);
                (vals, tmpVals) = (tmpVals, vals);
            }
        }
    }

    [BurstCompile]
    public struct BuildFlatGrid : IJob
    {
        public int particleCount;

        public NativeReference<int> UniqueCells;
        public NativeArray<int> uniqueCellKeys;

        [ReadOnly]  public NativeReference<int> totalCells;
        [ReadOnly]  public NativeArray<int> cellKey;
        [WriteOnly] public NativeArray<int> cellStart, cellEnd, cellIdToCompactIndex;
        public void Execute()
        {

            //Initialization 
            int val = cellKey[0];
            cellStart[0] = 0;
            uniqueCellKeys[0] = val;
            int it = 0;

            //Construct cellStart and cellEnd
            // we also count unique cells.
            for (int i = 1; i < particleCount; ++i)
            {
                if (cellKey[i] != val)
                {
                    cellEnd[it] = i;
                    it++;
                    val = cellKey[i];
                    uniqueCellKeys[it] = val;
                    cellStart[it] = i;
                }
            }
            cellEnd[it] = particleCount;
            UniqueCells.Value = it + 1;

            //cellIdToRun gives uniqueCell index from linear ID (-1 if it contains nothing)
            //uniqueCellKeys inverse map 
            int tc = math.min(totalCells.Value, cellIdToCompactIndex.Length);
            for (int i = 0; i < tc; i++) cellIdToCompactIndex[i] = -1;
            for (int u = 0; u < UniqueCells.Value; u++)
                cellIdToCompactIndex[uniqueCellKeys[u]] = u;

        }
    }

    [BurstCompile]
    public struct FindNeighSize : IJobParallelFor
    {
        public float invCellSize, h2;
        [ReadOnly] public NativeReference<int> sizeX, sizeY, sizeZ;
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<int3> neighborOffsets;
        [ReadOnly] public NativeReference<int> minCx, minCy, minCz;
        [ReadOnly] public NativeArray<int> cellIdToRun, cellStart, cellEnd;
        [WriteOnly] public NativeArray<int> nbrCount;
        public void Execute(int i)
        {
            int sx = sizeX.Value;
            int sy = sizeY.Value;
            int sz = sizeZ.Value;
            int mnCx = minCx.Value;
            int mnCy = minCy.Value;
            int mnCz = minCz.Value;


            int3 b = WorldToCell(positions[i], invCellSize);
            float3 pi = positions[i];

            int cnt = 0;
            for (int n = 0; n < neighborOffsets.Length; n++)
            {
                int3 c = neighborOffsets[n] + b;

                int lx = c.x - mnCx;
                int ly = c.y - mnCy;
                int lz = c.z - mnCz;

                //Check if it falls outside of the grid ( boundary )
                //Use unsigned int to check if lx<0 or lx >= sx faster
                // Works because negative numbers become large positive when cast to uint
                if ((uint)lx >= (uint)sx || (uint)ly >= (uint)sy || (uint)lz >= (uint)sz)
                    continue;

                int key = lx + sx * (ly + sy * lz);

                //Find index of cell 
                int compactCellIndex = cellIdToRun[key];

                // If compactCellIndex <0 cell is empty !
                if (compactCellIndex < 0)
                    continue;

                // Look trough n
                int s = cellStart[compactCellIndex], e = cellEnd[compactCellIndex];
                for (int j = s; j < e; j++)
                {
                    if (j == i)
                        continue; // skip self

                    float3 r = pi - positions[j];
                    if (math.lengthsq(r) < h2) cnt++;
                }
            }
            nbrCount[i] = cnt;


        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct constructNbrStartAndNbrEnd : IJob
    {
        public int particleCount;
        [ReadOnly] public NativeArray<int> nbrCount;
        [WriteOnly] public NativeArray<int> nbrStart, nbrEnd;
        public void Execute()
        {
            int total = 0;
            for (int i = 0; i < particleCount; i++)
            {
                nbrStart[i] = total;
                total += nbrCount[i];
                nbrEnd[i] = total;
            }

        }
    }


    [BurstCompile(FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Medium)]
    public struct ConstructNbrFlat : IJobParallelFor
    {
        public float invCellSize, h2;
        [ReadOnly] public NativeReference<int> sizeX, sizeY, sizeZ;
        [ReadOnly] public NativeArray<float3> positions;
        [ReadOnly] public NativeArray<int3> neighborOffsets;
        [ReadOnly] public NativeReference<int> minCx, minCy, minCz;
        [ReadOnly] public NativeArray<int> cellIdToRun, cellStart, cellEnd, nbrStart,nbrEnd;

        [NativeDisableParallelForRestriction]
        public NativeArray<int> nbrFlat;

        public void Execute(int i)
        {
            // Early out if this particle has no neighbors.
            if (nbrEnd[i] == nbrStart[i])
                return;

            // Cache all NativeReference values ONCE
            int sx = sizeX.Value;
            int sy = sizeY.Value;
            int sz = sizeZ.Value;
            int mnCx = minCx.Value;
            int mnCy = minCy.Value;
            int mnCz = minCz.Value;

            int3 b = WorldToCell(positions[i], invCellSize);
            float3 pi = positions[i];

            int baseIdx = nbrStart[i];
            int write = 0;

            for (int n = 0; n < neighborOffsets.Length; n++)
            {
                int3 c = neighborOffsets[n] + b;
                int lx = c.x - mnCx;
                int ly = c.y - mnCy;
                int lz = c.z - mnCz;


                //Check if it falls outside of the grid ( boundary )
                //Use unsigned int to check if lx<0 or lx >= sx faster
                if ((uint)lx >= (uint)sx || (uint)ly >= (uint)sy || (uint)lz >= (uint)sz)
                    continue;

                int key = lx + sx * (ly + sy * lz);
                int compactCellIndex = cellIdToRun[key];

                //Skip if cell is empty (-1)
                if (compactCellIndex < 0)
                    continue;

                int s = cellStart[compactCellIndex], e = cellEnd[compactCellIndex];
                for (int j = s; j < e; j++)
                {
                    //Skip self
                    if (j == i)
                        continue;

                    //We add the particle to the neighbor of the particle i
                    float3 r = pi - positions[j];
                    if (math.lengthsq(r) < h2)
                    {
                        nbrFlat[baseIdx + write] = j;
                        write++;
                    }
                }
            }

        }
    }
}