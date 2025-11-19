VERTEX_SHADER_SRC = """
#version 330 core
layout (location = 0) in vec2 a_position;
layout (location = 1) in vec2 a_texcoord;

out vec2 v_texcoord;

uniform vec2 u_scale;
uniform vec2 u_pan;

void main()
{
    vec2 scaled = a_position * u_scale + u_pan;
    gl_Position = vec4(scaled, 0.0, 1.0);
    v_texcoord = a_texcoord;
}
"""

FRAGMENT_SHADER_SRC = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;

uniform sampler2D u_image;
uniform float u_temperature;
uniform float u_tint;
uniform float u_exposure;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_vibrance;
uniform float u_highlights;
uniform float u_shadows;
uniform float u_whites;
uniform float u_blacks;

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

void main()
{
    vec2 uv = vec2(v_texcoord.x, 1.0 - v_texcoord.y);
    vec3 color = texture(u_image, uv).rgb;

    float lum = dot(color, LUMA);
    float L = lum;

    color *= exp2(u_exposure);
    color.r *= (1.0 + u_temperature);
    color.b *= (1.0 - u_temperature);
    color.g *= (1.0 - u_tint);
    color.rg += vec2(u_tint * 0.5, 0.0);
    color.b += u_tint * 0.5;

    color = (color - 0.5) * u_contrast + 0.5;

    float highlights = clamp((L - 0.5) / 0.5, 0.0, 1.0);
    float shadows = clamp((0.5 - L) / 0.5, 0.0, 1.0);
    float whites = clamp((L - 0.8) / 0.2, 0.0, 1.0);
    float blacks = clamp((0.2 - L) / 0.2, 0.0, 1.0);

    color += u_highlights * highlights;
    color += u_shadows * shadows;
    color += u_whites * whites;
    color += u_blacks * blacks;

    vec3 lum3 = vec3(lum);
    color = lum3 + (color - lum3) * u_saturation;

    float sat_dist = length(color - lum3);
    float vib_weight = 1.0 - clamp(sat_dist, 0.0, 1.0);
    color = lum3 + (color - lum3) * (1.0 + u_vibrance * vib_weight);

    color = clamp(color, 0.0, 1.0);
    fragColor = vec4(color, 1.0);
}
"""
