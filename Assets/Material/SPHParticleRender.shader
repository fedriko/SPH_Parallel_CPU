Shader "SPH/ParticleRender"
{
    Properties
    {
        _Color ("Color", Color) = (0.1, 0.5, 0.9, 1.0)
        _Radius ("Radius", Float) = 1.0
    }
    
    SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 5.0
            
            #include "UnityCG.cginc"
            
            float4 _Color;
            float _Radius;
            StructuredBuffer<float3> _positions;
            
            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };
            
            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 normal : TEXCOORD0;
            };
            
            v2f vert(appdata v, uint instanceID : SV_InstanceID)
            {
                v2f o;
                float3 worldPos = v.vertex.xyz * _Radius + _positions[instanceID];
                o.pos = UnityWorldToClipPos(float4(worldPos, 1.0));
                o.normal = v.normal;
                return o;
            }
            
            float4 frag(v2f i) : SV_Target
            {
                float3 light = normalize(float3(1, 1, -1));
                float diffuse = max(0.3, dot(normalize(i.normal), light));
                return _Color * diffuse;
            }
            ENDCG
        }
    }
}