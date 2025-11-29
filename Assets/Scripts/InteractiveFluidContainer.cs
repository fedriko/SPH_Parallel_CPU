using UnityEngine;

public class InteractiveFluidContainer : MonoBehaviour
{
    [Header("Container")]
    public Transform container;

    [Header("Movement Settings")]
    public float moveSpeed = 5f;
    public float rotationSpeed = 100f;
    public float smoothingSpeed = 5f;

    [Header("Limits")]
    public float maxMoveSpeedPerSecond = 2f;
    public float maxRotateSpeedPerSecond = 90f;

    private Camera mainCamera;
    private Vector3 targetPosition;
    private Quaternion targetRotation;
    private Vector3 physicsPosition;
    private Quaternion physicsRotation;

    private bool isDragging = false;
    private Vector3 dragOffset;
    private Plane dragPlane;

    void Start()
    {
        mainCamera = Camera.main;

        if (container == null)
        {
            Debug.LogError("Container not assigned!");
            return;
        }

        targetPosition = container.position;
        targetRotation = container.rotation;
        physicsPosition = container.position;
        physicsRotation = container.rotation;
    }

    void Update()
    {
        HandleInput();
        SmoothMovement();
    }

    void HandleInput()
    {
        // Left mouse button - drag to move
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit))
            {
                if (hit.transform == container)
                {
                    isDragging = true;
                    dragPlane = new Plane(Vector3.up, container.position);

                    float distance;
                    if (dragPlane.Raycast(ray, out distance))
                    {
                        Vector3 hitPoint = ray.GetPoint(distance);
                        dragOffset = container.position - hitPoint;
                    }
                }
            }
        }

        if (Input.GetMouseButtonUp(0))
        {
            isDragging = false;
        }

        // Drag movement
        if (isDragging)
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            float distance;
            if (dragPlane.Raycast(ray, out distance))
            {
                Vector3 hitPoint = ray.GetPoint(distance);
                targetPosition = hitPoint + dragOffset;
            }
        }

        // Right mouse button - rotate
        if (Input.GetMouseButton(1))
        {
            float rotX = Input.GetAxis("Mouse X") * rotationSpeed * Time.deltaTime;
            float rotY = Input.GetAxis("Mouse Y") * rotationSpeed * Time.deltaTime;

            targetRotation = Quaternion.Euler(0, rotX, 0) * targetRotation;
            targetRotation = targetRotation * Quaternion.Euler(rotY, 0, 0);
        }

        // Keyboard movement (WASDQE)
        Vector3 keyboardMove = Vector3.zero;
        if (Input.GetKey(KeyCode.W)) keyboardMove.z += 1f;
        if (Input.GetKey(KeyCode.S)) keyboardMove.z -= 1f;
        if (Input.GetKey(KeyCode.A)) keyboardMove.x -= 1f;
        if (Input.GetKey(KeyCode.D)) keyboardMove.x += 1f;
        if (Input.GetKey(KeyCode.Q)) keyboardMove.y -= 1f;
        if (Input.GetKey(KeyCode.E)) keyboardMove.y += 1f;

        if (keyboardMove != Vector3.zero)
        {
            targetPosition += keyboardMove.normalized * moveSpeed * Time.deltaTime;
        }
    }

    void SmoothMovement()
    {
        // Calculate desired change
        Vector3 deltaPos = targetPosition - physicsPosition;

        // Limit maximum movement speed
        float maxDelta = maxMoveSpeedPerSecond * Time.deltaTime;
        if (deltaPos.magnitude > maxDelta)
        {
            deltaPos = deltaPos.normalized * maxDelta;
        }

        // Apply smoothing
        physicsPosition += deltaPos * smoothingSpeed * Time.deltaTime;

        // Limit rotation speed
        float rotAngle = Quaternion.Angle(physicsRotation, targetRotation);
        float maxRotDelta = maxRotateSpeedPerSecond * Time.deltaTime;

        float t = smoothingSpeed * Time.deltaTime;
        if (rotAngle > maxRotDelta)
        {
            t = Mathf.Min(t, maxRotDelta / rotAngle);
        }

        physicsRotation = Quaternion.Slerp(physicsRotation, targetRotation, t);

        // Update actual container transform
        container.position = physicsPosition;
        container.rotation = physicsRotation;
    }

    // Public accessors for SPH collision system
    public Vector3 GetPhysicsPosition() => physicsPosition;
    public Quaternion GetPhysicsRotation() => physicsRotation;
}