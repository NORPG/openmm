/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 *                                                                            *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Ported from the OpenMM CUDA Platform.                                      *
 * Source: platforms/cuda/src/CudaProgram.cpp                                 *
 *                                                                            *
 * Original CUDA Platform code:                                               *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 *                                                                            *
 * Metal Platform code:                                                       *
 * Portions copyright (c) 2026 Chun-Chi Hung.                                 *
 * Authors: Chun-Chi Hung                                                     *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the               *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program. If not, see <http://www.gnu.org/licenses/>.       *
 * -------------------------------------------------------------------------- */

#include "MetalProgram.h"
#include "MetalContext.h"
#include "MetalKernel.h"
#import <Metal/Metal.h>

using namespace OpenMM;
using namespace std;

struct MetalProgram::Impl {
    id<MTLLibrary> library;
};

MetalProgram::MetalProgram(MetalContext& context, void* library) : impl(new Impl()), context(context) {
    impl->library = (__bridge id<MTLLibrary>) library;
}

MetalProgram::~MetalProgram() {
}

ComputeKernel MetalProgram::createKernel(const string& name) {
    @autoreleasepool {
        NSString* functionName = [NSString stringWithUTF8String:name.c_str()];
        if (![impl->library.functionNames containsObject:functionName])
            throw OpenMMException("Unknown Metal kernel: "+name);
        MTL4LibraryFunctionDescriptor* function = [MTL4LibraryFunctionDescriptor new];
        function.library = impl->library;
        function.name = functionName;
        MTL4ComputePipelineDescriptor* descriptor = [MTL4ComputePipelineDescriptor new];
        descriptor.computeFunctionDescriptor = function;
        id<MTL4Compiler> compiler = (__bridge id<MTL4Compiler>) context.getCompiler();
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline = [compiler newComputePipelineStateWithDescriptor:descriptor
                compilerTaskOptions:nil error:&error];
        if (pipeline == nil)
            throw OpenMMException("Error creating Metal pipeline "+name+": "+
                    (error == nil ? string("unknown error") : string(error.localizedDescription.UTF8String)));
        return ComputeKernel(new MetalKernel(context, (__bridge void*) pipeline, name));
    }
}
