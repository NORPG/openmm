/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Adapted from Common pairwise kernels used by CUDA, HIP, and OpenCL.        *
 * Original OpenMM code:                                                      *
 * Portions copyright (c) 2008-2026 Stanford University and the Authors.       *
 * Authors: Peter Eastman                                                     *
 * Metal Platform code:                                                       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                 *
 * Authors: Chun-Chi Hung                                                     *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 * This program is distributed WITHOUT ANY WARRANTY; without even the        *
 * implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. *
 * See the GNU Lesser General Public License for more details.                *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 * -------------------------------------------------------------------------- */

#include "MetalPairwiseOptimizations.h"
#include "openmm/OpenMMException.h"
#include <cctype>
#include <regex>
#include <sstream>

using namespace OpenMM;
using namespace std;

namespace {

/** Validate identifiers supplied by an explicit parameter metadata declaration. */
bool metadataIdentifier(const string& name) {
    if (name.empty() || (!isalpha(static_cast<unsigned char>(name[0])) && name[0] != '_'))
        return false;
    for (char character : name)
        if (!isalnum(static_cast<unsigned char>(character)) && character != '_')
            return false;
    return true;
}

string define(const string& name, bool enabled) {
    return "#define "+name+(enabled ? " 1\n" : " 0\n");
}

/**
 * Read Common's generated field declarations without changing its source.
 *
 * Metal's template retains those declarations in its baseline branch.  The
 * register branch needs their types and names to declare, broadcast, and rotate
 * lane-owned records.  Only this explicitly marked declaration list is read;
 * unknown declarations fail rather than silently omitting transported state.
 */
string customGBFieldDefines(const string& source) {
    const string beginMarker = "// OPENMM_METAL_CUSTOM_GB_FIELDS_BEGIN\n";
    const string endMarker = "// OPENMM_METAL_CUSTOM_GB_FIELDS_END";
    const size_t begin = source.find(beginMarker);
    if (begin == string::npos)
        return "";
    const size_t end = source.find(endMarker, begin+beginMarker.size());
    if (end == string::npos || source.find(beginMarker, begin+beginMarker.size()) != string::npos)
        throw OpenMMException("Invalid Metal CustomGB field metadata markers");
    const string fields = source.substr(begin+beginMarker.size(), end-begin-beginMarker.size());
    string readonly, accumulators;
    set<string> names;
    const regex declaration("\\s*LOCAL ([A-Za-z0-9_]+) (local_[A-Za-z0-9_]+)\\[LOCAL_BUFFER_SIZE\\];");
    size_t cursor = 0;
    while (cursor < fields.size()) {
        smatch match;
        const string remaining = fields.substr(cursor);
        if (!regex_search(remaining, match, declaration, regex_constants::match_continuous)) {
            // An unexpanded template is inspected by source-only tests.
            if (regex_match(remaining, regex("\\s*(ATOM_PARAMETER_DATA)?\\s*")))
                break;
            throw OpenMMException("Unrecognized Metal CustomGB field declaration: "+remaining);
        }
        const string type = match[1], name = match[2];
        if (!names.insert(name).second)
            throw OpenMMException("Duplicate Metal CustomGB lane field: "+name);
        const string entry = " OP("+type+", "+name+")";
        if (name.find("local_params") == 0 || name.find("local_values") == 0)
            readonly += entry;
        else if (name.find("local_deriv") == 0 || name.find("local_dValue0dParam") == 0)
            accumulators += entry;
        else
            throw OpenMMException("Unrecognized Metal CustomGB lane field: "+name);
        cursor += match.length();
    }
    return "#define CUSTOM_GB_READONLY_FIELDS(OP)"+readonly+"\n"+
            "#define CUSTOM_GB_ACCUMULATOR_FIELDS(OP)"+accumulators+"\n";
}

} // namespace

MetalPairwiseOptimizations::Settings MetalPairwiseOptimizations::getBuildSettings() {
    Settings settings;
#if OPENMM_METAL_FAST_CUSTOM_GB_VALUE_SHUFFLE
    settings.customGBValue = true;
#endif
#if OPENMM_METAL_FAST_CUSTOM_GB_ENERGY_SHUFFLE
    settings.customGBEnergy = true;
#endif
#if OPENMM_METAL_FAST_GBSA_BORN_SHUFFLE
    settings.gbsaBorn = true;
#endif
#if OPENMM_METAL_FAST_GBSA_FORCE_SHUFFLE
    settings.gbsaForce = true;
#endif
#if OPENMM_METAL_FAST_DPD_PARTICLE_SHUFFLE
    settings.dpdParticles = true;
#endif
#if OPENMM_METAL_FAST_DPD_TILE_BROADCAST
    settings.dpdTile = true;
#endif
#if OPENMM_METAL_FAST_CUSTOM_HBOND_SHUFFLE
    settings.customHbond = true;
#endif
    return settings;
}

string MetalPairwiseOptimizations::apply(const string& source) {
    return apply(source, getBuildSettings());
}

set<string> MetalPairwiseOptimizations::getGBSAChainRuleBornForceParameters(const string& source) {
    istringstream lines(source);
    string line;
    set<string> parameters;
    while (getline(lines, line)) {
        istringstream declaration(line);
        string comment, marker, first, second, extra;
        if (!(declaration >> comment >> marker) || comment != "//" ||
                marker != "OPENMM_METAL_GBSA_BORN_FORCE_PARAMETERS")
            continue;
        // Read only the source-owned declaration; formulas are never inspected.
        if (!parameters.empty() || !(declaration >> first >> second) || (declaration >> extra) ||
                !metadataIdentifier(first) || !metadataIdentifier(second) ||
                first.size() < 2 || second.size() < 2 || first.back() != '1' || second.back() != '2' ||
                first.substr(0, first.size()-1) != second.substr(0, second.size()-1))
            return {};
        parameters = {first, second};
    }
    return parameters;
}

string MetalPairwiseOptimizations::apply(const string& source, const Settings& settings) {
    // No kernel recognition is needed: each shader owns the branches it uses.
    // Preserve explicit test overrides when the adapter applies build defaults.
    const string marker = "#define OPENMM_METAL_PAIRWISE_OPTIONS_DEFINED\n";
    if (source.find(marker) != string::npos)
        return source;
    return marker+define("USE_CUSTOM_GB_VALUE_SHUFFLE", settings.customGBValue)+
            define("USE_CUSTOM_GB_ENERGY_SHUFFLE", settings.customGBEnergy)+
            define("USE_GBSA_BORN_SHUFFLE", settings.gbsaBorn)+
            define("USE_GBSA_FORCE_SHUFFLE", settings.gbsaForce)+
            define("USE_DPD_PARTICLE_SHUFFLE", settings.dpdParticles)+
            define("USE_DPD_TILE_BROADCAST", settings.dpdTile)+
            define("USE_CUSTOM_HBOND_SHUFFLE", settings.customHbond)+customGBFieldDefines(source)+source;
}
