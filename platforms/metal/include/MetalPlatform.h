#ifndef OPENMM_METALPLATFORM_H_
#define OPENMM_METALPLATFORM_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2026 Stanford University and the Authors.           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/Platform.h"
#include "windowsExportMetal.h"

namespace OpenMM {

class MetalContext;

/**
 * OpenMM platform for native Metal compute on Apple Silicon.
 *
 * The initial backend supports one built-in Apple GPU and single precision.
 * Metal framework types are intentionally absent from this public C++ header.
 */
class OPENMM_EXPORT_METAL MetalPlatform : public Platform {
public:
    class PlatformData;

    MetalPlatform();

    const std::string& getName() const {
        static const std::string name = "Metal";
        return name;
    }
    double getSpeed() const;
    bool supportsDoublePrecision() const;

    /** Return whether this process has a supported built-in Apple GPU. */
    static bool isPlatformSupported();

    /** Return the native state owned by an OpenMM Context using this platform. */
    static MetalContext& getMetalContext(ContextImpl& context);
    static const MetalContext& getMetalContext(const ContextImpl& context);

    const std::string& getPropertyValue(const Context& context, const std::string& property) const;
    std::vector<std::map<std::string, std::string> > getDevices(const std::map<std::string, std::string>& filters={}) const;
    void contextCreated(ContextImpl& context, const std::map<std::string, std::string>& properties) const;
    void contextDestroyed(ContextImpl& context) const;

    /** Select the sole GPU exposed by this initial backend. */
    static const std::string& MetalDeviceIndex() {
        static const std::string key = "DeviceIndex";
        return key;
    }

    /** Report the selected Metal device name. */
    static const std::string& MetalDeviceName() {
        static const std::string key = "DeviceName";
        return key;
    }

    /** Select numerical precision.  The only accepted value is "single". */
    static const std::string& MetalPrecision() {
        static const std::string key = "Precision";
        return key;
    }
};

} // namespace OpenMM

#endif // OPENMM_METALPLATFORM_H_
