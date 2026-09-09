// OPENMM_METAL_NATIVE_BODY
/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * Metal Platform code: copyright (c) 2026 Chun-Chi Hung.                       *
 * Author: Chun-Chi Hung.                                                      *
 * This program is free software under the GNU Lesser General Public License, *
 * version 3 or (at your option) any later version, without any warranty.       *
 * See <http://www.gnu.org/licenses/> for the license.                          *
 * -------------------------------------------------------------------------- */

/**
 * @brief Round nonnegative bounds upward so FP16 boxes never shrink.
 * Only a rounded-down value advances; exact values and infinity stay unchanged.
 * Older language targets retain the equivalent positive-half bit increment.
 */
inline half metalBoundHalf(float value) {
    half result = half(value);
    ushort bits = as_type<ushort>(result);
    if (float(result) < value) result = as_type<half>(ushort(bits+1));
    return result;
}

/** @brief Compress the three bounding-box half widths; the unused lane is zero. */
inline half4 metalBoundsHalf(float4 value) {
    return half4(metalBoundHalf(value.x), metalBoundHalf(value.y), metalBoundHalf(value.z), half(0));
}
