/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Native Metal runtime implemented in Swift.  OpenMM's C++ implementation   *
 * reaches this file exclusively through the C ABI in MetalSwiftBridge.h.     *
 * -------------------------------------------------------------------------- */

@preconcurrency import Foundation
@preconcurrency import Metal
import Dispatch
import Darwin

private let bridgeSuccess: Int32 = 0
private let bridgeInvalidArgument: Int32 = 1
private let bridgeOutOfRange: Int32 = 2
private let bridgeUnsupported: Int32 = 3
private let bridgeNativeError: Int32 = 4
private let bridgeInvalidUTF8: Int32 = 5
private let bridgeInternalError: Int32 = 255

private struct BridgeFailure: Error {
    let code: Int32
    let message: String
}

private func failure(_ code: Int32, _ message: String) -> BridgeFailure {
    BridgeFailure(code: code, message: message)
}

private func errorDescription(_ error: Error) -> String {
    if let bridge = error as? BridgeFailure {
        return bridge.message
    }
    let native = error as NSError
    var result = native.localizedDescription
    if let reason = native.localizedFailureReason, !reason.isEmpty, reason != result {
        result += ": \(reason)"
    }
    return result.isEmpty ? String(describing: error) : result
}

private func prepareError(_ output: UnsafeMutablePointer<OMMMetalErrorC>?) {
    output?.pointee.code = bridgeSuccess
    output?.pointee.message = nil
}

private func reportError(_ error: Error, to output: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    let code = (error as? BridgeFailure)?.code ?? bridgeNativeError
    output?.pointee.code = code
    output?.pointee.message = errorDescription(error).withCString { strdup($0) }
    return code
}

@discardableResult
private func bridgeCall(_ errorOutput: UnsafeMutablePointer<OMMMetalErrorC>?,
                        _ body: () throws -> Void) -> Int32 {
    prepareError(errorOutput)
    do {
        try autoreleasepool(invoking: body)
        return bridgeSuccess
    }
    catch {
        return reportError(error, to: errorOutput)
    }
}

private func decodeUTF8(_ pointer: UnsafePointer<CChar>?, _ length: Int,
                        _ description: String, allowEmpty: Bool = true) throws -> String {
    if length == 0 {
        if allowEmpty {
            return ""
        }
        throw failure(bridgeInvalidArgument, "\(description) must not be empty")
    }
    guard let pointer else {
        throw failure(bridgeInvalidArgument, "\(description) is null")
    }
    let bytes = UnsafeRawBufferPointer(start: pointer, count: length)
    guard let result = String(bytes: bytes, encoding: .utf8) else {
        throw failure(bridgeInvalidUTF8, "\(description) is not valid UTF-8")
    }
    return result
}

private func checkedInt(_ value: UInt, _ description: String) throws -> Int {
    guard value <= UInt(Int.max) else {
        throw failure(bridgeOutOfRange, "\(description) exceeds the native address range")
    }
    return Int(value)
}

private func bridgeObject<T: AnyObject>(_ handle: UnsafeMutableRawPointer?,
                                         as type: T.Type, _ description: String) throws -> T {
    guard let handle else {
        throw failure(bridgeInvalidArgument, "\(description) is null")
    }
    return Unmanaged<T>.fromOpaque(handle).takeUnretainedValue()
}

private extension NSLock {
    func withBridgeLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

private struct DeviceSnapshot {
    let name: String
    let registryId: UInt64
    let lowPower: Bool
    let removable: Bool
    let unifiedMemory: Bool
    let maxBufferLength: UInt64
    let recommendedMaxWorkingSetSize: UInt64
    let maxThreadgroupMemoryLength: UInt64
    let maxThreadsPerThreadgroup: UInt64
    let argumentBufferTier: Int32
    let highestAppleGpuFamily: Int32
    let mac2: Bool
    let metal3: Bool
    let metal4: Bool

    init(device: any MTLDevice) {
        name = device.name
        registryId = device.registryID
        lowPower = device.isLowPower
        removable = device.isRemovable
        unifiedMemory = device.hasUnifiedMemory
        maxBufferLength = UInt64(device.maxBufferLength)
        recommendedMaxWorkingSetSize = UInt64(device.recommendedMaxWorkingSetSize)
        maxThreadgroupMemoryLength = UInt64(device.maxThreadgroupMemoryLength)
        maxThreadsPerThreadgroup = UInt64(device.maxThreadsPerThreadgroup.width)
        argumentBufferTier = device.argumentBuffersSupport == .tier2 ? 2 : 1

        var highest: Int32 = 0
        for family in 1...10 {
            if let nativeFamily = MTLGPUFamily(rawValue: 1000 + family),
               device.supportsFamily(nativeFamily) {
                highest = Int32(family)
            }
        }
        highestAppleGpuFamily = highest
        mac2 = device.supportsFamily(.mac2)
        metal3 = device.supportsFamily(.metal3)
        if #available(macOS 26.0, *) {
            metal4 = device.supportsFamily(.metal4)
        }
        else {
            metal4 = false
        }
    }
}

private func zeroCaps(_ output: UnsafeMutablePointer<OMMMetalDeviceCapsC>) {
    memset(output, 0, MemoryLayout<OMMMetalDeviceCapsC>.size)
}

private func writeCaps(_ snapshot: DeviceSnapshot,
                       to output: UnsafeMutablePointer<OMMMetalDeviceCapsC>) throws {
    zeroCaps(output)
    guard let name = snapshot.name.withCString({ strdup($0) }) else {
        throw failure(bridgeInternalError, "Unable to allocate a Metal device name")
    }
    output.pointee.registryId = snapshot.registryId
    output.pointee.maxBufferLength = snapshot.maxBufferLength
    output.pointee.recommendedMaxWorkingSetSize = snapshot.recommendedMaxWorkingSetSize
    output.pointee.maxThreadgroupMemoryLength = snapshot.maxThreadgroupMemoryLength
    output.pointee.maxThreadsPerThreadgroup = snapshot.maxThreadsPerThreadgroup
    output.pointee.argumentBufferTier = snapshot.argumentBufferTier
    output.pointee.highestAppleGpuFamily = snapshot.highestAppleGpuFamily
    output.pointee.lowPower = snapshot.lowPower ? 1 : 0
    output.pointee.removable = snapshot.removable ? 1 : 0
    output.pointee.unifiedMemory = snapshot.unifiedMemory ? 1 : 0
    output.pointee.mac2 = snapshot.mac2 ? 1 : 0
    output.pointee.metal3 = snapshot.metal3 ? 1 : 0
    output.pointee.metal4 = snapshot.metal4 ? 1 : 0
    output.pointee.name = name
}

private func isSupportedDefaultDevice(_ device: (any MTLDevice)?) -> Bool {
#if arch(arm64)
    guard let device else {
        return false
    }
    return device.hasUnifiedMemory && !device.isRemovable &&
           device.supportsFamily(.apple1) && device.supportsFamily(.metal3)
#else
    return false
#endif
}

private final class RetainedResources: @unchecked Sendable {
    let values: [AnyObject]

    init(_ values: [AnyObject]) {
        self.values = values
    }
}

private final class QueueBox: @unchecked Sendable {
    let device: any MTLDevice
    let commandQueue: any MTLCommandQueue
    let caps: DeviceSnapshot
    let submissionLock = NSLock()

    private let statusCondition = NSCondition()
    private var pendingAsyncCommands = 0
    private var asynchronousErrors: [String] = []

    init(deviceIndex: Int) throws {
        let devices = MTLCopyAllDevices()
        guard deviceIndex >= 0 && deviceIndex < devices.count else {
            throw failure(bridgeOutOfRange,
                          "Cannot create Metal queue: device index \(deviceIndex) is out of range (\(devices.count) devices available)")
        }
        let selected = devices[deviceIndex]
        guard let queue = selected.makeCommandQueue() else {
            throw failure(bridgeNativeError,
                          "Cannot create a Metal command queue for device \(selected.name)")
        }
        device = selected
        commandQueue = queue
        caps = DeviceSnapshot(device: selected)
        queue.label = "OpenMM Metal queue"
    }

    init(siblingOf parent: QueueBox) throws {
        guard let queue = parent.device.makeCommandQueue() else {
            throw failure(bridgeNativeError,
                          "Cannot create a sibling Metal command queue for device \(parent.caps.name)")
        }
        device = parent.device
        commandQueue = queue
        caps = parent.caps
        queue.label = "OpenMM Metal sibling queue"
    }

    func makeCommandBuffer(_ label: String) throws -> any MTLCommandBuffer {
        guard let command = commandQueue.makeCommandBuffer() else {
            throw failure(bridgeNativeError,
                          "Metal failed to allocate a command buffer for \(label)")
        }
        command.label = label
        return command
    }

    private func commandFailure(_ command: any MTLCommandBuffer) -> BridgeFailure {
        let detail = command.error.map(errorDescription) ?? "unknown Metal error"
        return failure(bridgeNativeError,
                       "Metal command '\(command.label ?? "unnamed")' failed: \(detail)")
    }

    func submitLocked(_ command: any MTLCommandBuffer, blocking: Bool,
                      retainedResources: [AnyObject] = [],
                      completion: (@Sendable (Bool) -> Void)? = nil) throws {
        let retained = RetainedResources(retainedResources)
        if blocking {
            command.commit()
            command.waitUntilCompleted()
            let succeeded = command.status == .completed
            completion?(succeeded)
            withExtendedLifetime(retained) {}
            if !succeeded {
                throw commandFailure(command)
            }
            return
        }

        statusCondition.lock()
        pendingAsyncCommands += 1
        statusCondition.unlock()

        command.addCompletedHandler { [self, retained] completed in
            let succeeded = completed.status == .completed
            completion?(succeeded)
            if !succeeded {
                recordError(commandFailure(completed).message)
            }
            withExtendedLifetime(retained) {}
            statusCondition.lock()
            pendingAsyncCommands -= 1
            statusCondition.broadcast()
            statusCondition.unlock()
        }
        command.commit()
    }

    func recordError(_ message: String) {
        statusCondition.lock()
        asynchronousErrors.append(message)
        statusCondition.unlock()
    }

    func checkForErrors() throws {
        statusCondition.lock()
        let errors = asynchronousErrors
        asynchronousErrors.removeAll(keepingCapacity: true)
        statusCondition.unlock()
        if !errors.isEmpty {
            throw failure(bridgeNativeError, errors.joined(separator: "\n"))
        }
    }

    func waitUntilIdle() throws {
        var markerError: Error?
        do {
            try submissionLock.withBridgeLock {
                let marker = try makeCommandBuffer("wait until idle")
                try submitLocked(marker, blocking: true)
            }
        }
        catch {
            markerError = error
        }

        // Even a failed marker must not let asynchronous host work escape this
        // synchronization boundary.  In particular, a completed download may
        // still be copying from its staging buffer on a completion callback.
        statusCondition.lock()
        while pendingAsyncCommands != 0 {
            statusCondition.wait()
        }
        statusCondition.unlock()

        var pendingError: Error?
        do {
            try checkForErrors()
        }
        catch {
            pendingError = error
        }
        if let markerError, let pendingError {
            throw failure(bridgeNativeError,
                          errorDescription(markerError)+"\n"+errorDescription(pendingError))
        }
        if let markerError {
            throw markerError
        }
        if let pendingError {
            throw pendingError
        }
    }
}

private final class BufferBox: @unchecked Sendable {
    let queue: QueueBox
    let name: String
    private let stateLock = NSLock()
    private var buffer: any MTLBuffer
    private var logicalLength: Int

    init(queue: QueueBox, bytes: Int, name: String) throws {
        self.queue = queue
        self.name = name
        guard let created = queue.device.makeBuffer(length: max(bytes, 1),
                                                     options: .storageModePrivate) else {
            throw failure(bridgeNativeError,
                          "Error allocating Metal array \(name) (\(bytes) bytes)")
        }
        created.label = name
        buffer = created
        logicalLength = bytes
    }

    func validate(offset: Int, bytes: Int, operation: String) throws {
        let length = stateLock.withBridgeLock { logicalLength }
        guard offset >= 0 && bytes >= 0 && offset <= length && bytes <= length - offset else {
            throw failure(bridgeOutOfRange, "\(operation): requested range exceeds the Metal buffer")
        }
    }

    func snapshot() -> any MTLBuffer {
        stateLock.withBridgeLock { buffer }
    }

    func resize(_ bytes: Int) throws {
        try queue.checkForErrors()
        try queue.submissionLock.withBridgeLock {
            guard let replacement = queue.device.makeBuffer(length: max(bytes, 1),
                                                            options: .storageModePrivate) else {
                throw failure(bridgeNativeError,
                              "Error allocating Metal array \(name) (\(bytes) bytes)")
            }
            replacement.label = name
            stateLock.withBridgeLock {
                buffer = replacement
                logicalLength = bytes
            }
        }
    }
}

private final class DownloadRequest: @unchecked Sendable {
    let destination: UnsafeMutableRawPointer
    let staging: any MTLBuffer
    let byteCount: Int

    init(destination: UnsafeMutableRawPointer, staging: any MTLBuffer, byteCount: Int) {
        self.destination = destination
        self.staging = staging
        self.byteCount = byteCount
    }

    func complete(_ succeeded: Bool) {
        if succeeded {
            memcpy(destination, staging.contents(), byteCount)
        }
    }
}

private final class ProgramBox: @unchecked Sendable {
    let queue: QueueBox
    let library: any MTLLibrary

    init(queue: QueueBox, source: String, fastMath: Bool,
         languageMajor: UInt16, languageMinor: UInt16) throws {
        guard !source.isEmpty else {
            throw failure(bridgeInvalidArgument, "Cannot compile an empty Metal program")
        }
        let options = MTLCompileOptions()
        options.fastMathEnabled = fastMath
        if languageMajor != 0 {
            let encoded = (UInt(languageMajor) << 16) | UInt(languageMinor)
            guard let version = MTLLanguageVersion(rawValue: encoded) else {
                throw failure(bridgeInvalidArgument,
                              "Unsupported Metal language version \(languageMajor).\(languageMinor)")
            }
            options.languageVersion = version
        }
        self.queue = queue
        library = try queue.device.makeLibrary(source: source, options: options)
    }

    init(queue: QueueBox, libraryBytes: UnsafeRawPointer, length: Int) throws {
        guard length > 0 else {
            throw failure(bridgeInvalidArgument, "Cannot load an empty Metal library")
        }
        let copied = Data(bytes: libraryBytes, count: length)
        let dispatchData = copied.withUnsafeBytes { DispatchData(bytes: $0) }
        self.queue = queue
        library = try queue.device.makeLibrary(data: dispatchData)
    }
}

private enum KernelArgument: @unchecked Sendable {
    case empty
    case buffer(BufferBox)
    case bytes(Data)
}

private enum ArgumentBufferSlot {
    case buffer
    case constant(Int)
    case unsupported(String)
}

/** Return the in-argument-buffer storage size for non-aggregate MSL values. */
private func argumentConstantSize(_ type: MTLDataType) -> Int? {
    switch type {
    case .float, .int, .uint:
        return 4
    case .float2, .int2, .uint2:
        return 8
    case .float3, .float4, .int3, .int4, .uint3, .uint4:
        return 16
    case .half, .short, .ushort:
        return 2
    case .half2, .short2, .ushort2:
        return 4
    case .half3, .half4, .short3, .short4, .ushort3, .ushort4:
        return 8
    case .char, .uchar, .bool:
        return 1
    case .char2, .uchar2, .bool2:
        return 2
    case .char3, .char4, .uchar3, .uchar4, .bool3, .bool4:
        return 4
    case .long, .ulong:
        return 8
    case .long2, .ulong2:
        return 16
    case .long3, .long4, .ulong3, .ulong4:
        return 32
    default:
        // Aggregates, matrices, resources, and future SDK types require a
        // dedicated encoder instead of guessing their ABI layout.
        return nil
    }
}

private final class KernelBox: @unchecked Sendable {
    let queue: QueueBox
    let function: any MTLFunction
    let pipeline: any MTLComputePipelineState
    let argumentEncoder: (any MTLArgumentEncoder)?
    let name: String
    let bindingMode: Int32
    let argumentBufferIndex: Int
    let argumentBufferSlots: [Int: ArgumentBufferSlot]

    private let stateLock = NSLock()
    private var arguments: [KernelArgument] = []

    init(program: ProgramBox, name: String, bindingMode: Int32,
         argumentBufferIndex: Int) throws {
        guard argumentBufferIndex >= 0 else {
            throw failure(bridgeInvalidArgument,
                          "Metal argument buffer index must not be negative")
        }
        guard bindingMode == 0 || bindingMode == 1 else {
            throw failure(bridgeInvalidArgument, "Unknown Metal kernel binding mode")
        }
        guard let function = program.library.makeFunction(name: name) else {
            throw failure(bridgeInvalidArgument,
                          "Metal program does not contain a kernel named '\(name)'")
        }
        let pipeline: any MTLComputePipelineState
        var slots: [Int: ArgumentBufferSlot] = [:]
        if bindingMode == 1 {
            var reflection: MTLComputePipelineReflection?
            pipeline = try program.queue.device.makeComputePipelineState(
                function: function,
                options: [.bindingInfo, .bufferTypeInfo],
                reflection: &reflection)
            guard let binding = reflection?.bindings.first(where: {
                      $0.index == argumentBufferIndex
                  }),
                  let bufferBinding = binding as? any MTLBufferBinding,
                  bufferBinding.bufferPointerType?.elementIsArgumentBuffer == true,
                  let structure = bufferBinding.bufferStructType else {
                throw failure(bridgeInvalidArgument,
                              "Kernel \(name) has no reflected argument buffer at buffer index \(argumentBufferIndex)")
            }
            for member in structure.members {
                let index = member.argumentIndex
                if member.dataType == .pointer {
                    slots[index] = .buffer
                }
                else if let byteCount = argumentConstantSize(member.dataType) {
                    slots[index] = .constant(byteCount)
                }
                else {
                    slots[index] = .unsupported(String(describing: member.dataType))
                }
            }
        }
        else {
            pipeline = try program.queue.device.makeComputePipelineState(function: function)
        }
        var encoder: (any MTLArgumentEncoder)?
        if bindingMode == 1 {
            encoder = function.makeArgumentEncoder(bufferIndex: argumentBufferIndex)
        }
        if bindingMode == 1 && encoder == nil {
            throw failure(bridgeInvalidArgument,
                          "Kernel \(name) has no argument buffer at buffer index \(argumentBufferIndex)")
        }
        queue = program.queue
        self.function = function
        self.pipeline = pipeline
        argumentEncoder = encoder
        self.name = name
        self.bindingMode = bindingMode
        self.argumentBufferIndex = argumentBufferIndex
        argumentBufferSlots = slots
    }

    private func validateBufferSlot(_ index: Int) throws {
        guard bindingMode == 1 else { return }
        guard let slot = argumentBufferSlots[index] else {
            throw failure(bridgeInvalidArgument,
                          "Argument \(index) is not present in the reflected argument buffer for kernel \(name)")
        }
        guard case .buffer = slot else {
            throw failure(bridgeInvalidArgument,
                          "Argument \(index) is not a buffer slot in the argument buffer for kernel \(name)")
        }
    }

    private func validateConstantSlot(_ index: Int, byteCount: Int) throws {
        guard bindingMode == 1 else { return }
        guard let slot = argumentBufferSlots[index] else {
            throw failure(bridgeInvalidArgument,
                          "Argument \(index) is not present in the reflected argument buffer for kernel \(name)")
        }
        switch slot {
        case .constant(let expected) where expected == byteCount:
            return
        case .constant(let expected):
            throw failure(bridgeInvalidArgument,
                          "Argument \(index) for kernel \(name) requires \(expected) bytes, not \(byteCount)")
        case .buffer:
            throw failure(bridgeInvalidArgument,
                          "Argument \(index) is a buffer slot, not a constant slot, in kernel \(name)")
        case .unsupported(let type):
            throw failure(bridgeUnsupported,
                          "Argument \(index) for kernel \(name) has unsupported reflected type \(type)")
        }
    }

    func addBuffer(_ buffer: BufferBox) throws {
        guard buffer.queue === queue else {
            throw failure(bridgeInvalidArgument,
                          "Cannot bind a Metal buffer from a different command queue to kernel \(name)")
        }
        try stateLock.withBridgeLock {
            try validateBufferSlot(arguments.count)
            arguments.append(.buffer(buffer))
        }
    }

    func addBytes(_ value: UnsafeRawPointer, size: Int) throws {
        let data = Data(bytes: value, count: size)
        try stateLock.withBridgeLock {
            try validateConstantSlot(arguments.count, byteCount: size)
            arguments.append(.bytes(data))
        }
    }

    func addEmpty() {
        stateLock.withBridgeLock { arguments.append(.empty) }
    }

    func setBuffer(index: Int, buffer: BufferBox) throws {
        guard buffer.queue === queue else {
            throw failure(bridgeInvalidArgument,
                          "Cannot bind a Metal buffer from a different command queue to kernel \(name)")
        }
        try stateLock.withBridgeLock {
            guard index >= 0 && index < arguments.count else {
                throw failure(bridgeOutOfRange,
                              "Invalid argument index for Metal kernel \(name)")
            }
            try validateBufferSlot(index)
            arguments[index] = .buffer(buffer)
        }
    }

    func setBytes(index: Int, value: UnsafeRawPointer, size: Int) throws {
        let data = Data(bytes: value, count: size)
        try stateLock.withBridgeLock {
            guard index >= 0 && index < arguments.count else {
                throw failure(bridgeOutOfRange,
                              "Invalid argument index for Metal kernel \(name)")
            }
            try validateConstantSlot(index, byteCount: size)
            arguments[index] = .bytes(data)
        }
    }

    func maxBlockSize() -> Int {
        pipeline.maxTotalThreadsPerThreadgroup
    }

    func execute(threads: Int, requestedBlockSize: Int) throws {
        guard threads >= 0 else {
            throw failure(bridgeInvalidArgument,
                          "Cannot execute Metal kernel \(name) with a negative thread count")
        }
        if threads == 0 {
            return
        }
        let maximum = maxBlockSize()
        var blockSize = requestedBlockSize
        if blockSize == -1 {
            let width = max(1, pipeline.threadExecutionWidth)
            blockSize = min(256, maximum)
            blockSize = max(width, (blockSize / width) * width)
            blockSize = min(blockSize, maximum)
        }
        guard blockSize > 0 && blockSize <= maximum else {
            throw failure(bridgeInvalidArgument,
                          "Invalid threadgroup size \(blockSize) for Metal kernel \(name) (maximum \(maximum))")
        }

        try stateLock.withBridgeLock {
            if arguments.contains(where: {
                if case .empty = $0 { return true }
                return false
            }) {
                throw failure(bridgeInvalidArgument,
                              "At least one argument has not been set for Metal kernel \(name)")
            }

            try queue.checkForErrors()
            try queue.submissionLock.withBridgeLock {
                let command = try queue.makeCommandBuffer("execute \(name)")
                guard let encoder = command.makeComputeCommandEncoder() else {
                    throw failure(bridgeNativeError,
                                  "Metal failed to create a compute encoder for kernel \(name)")
                }
                encoder.label = name
                encoder.setComputePipelineState(pipeline)
                var resources: [AnyObject] = []

                if bindingMode == 0 {
                    for (index, argument) in arguments.enumerated() {
                        switch argument {
                        case .empty:
                            break
                        case .buffer(let box):
                            let native = box.snapshot()
                            encoder.setBuffer(native, offset: 0, index: index)
                            resources.append(native as AnyObject)
                        case .bytes(let data):
                            data.withUnsafeBytes { bytes in
                                encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: index)
                            }
                        }
                    }
                }
                else {
                    guard let argumentEncoder else {
                        throw failure(bridgeInternalError,
                                      "Metal argument encoder is missing for kernel \(name)")
                    }
                    guard let argumentBuffer = queue.device.makeBuffer(
                        length: max(argumentEncoder.encodedLength, 1),
                        options: .storageModeShared) else {
                        throw failure(bridgeNativeError,
                                      "Unable to allocate argument buffer for Metal kernel \(name)")
                    }
                    argumentBuffer.label = "OpenMM Metal encoded arguments"
                    argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
                    resources.append(argumentBuffer as AnyObject)
                    for (index, argument) in arguments.enumerated() {
                        switch argument {
                        case .empty:
                            break
                        case .buffer(let box):
                            let native = box.snapshot()
                            argumentEncoder.setBuffer(native, offset: 0, index: index)
                            encoder.useResource(native, usage: [.read, .write])
                            resources.append(native as AnyObject)
                        case .bytes(let data):
                            let destination = argumentEncoder.constantData(at: index)
                            let baseAddress = argumentBuffer.contents()
                            let destinationAddress = UInt(bitPattern: destination)
                            let bufferAddress = UInt(bitPattern: baseAddress)
                            let encodedLength = UInt(argumentEncoder.encodedLength)
                            guard destinationAddress >= bufferAddress else {
                                throw failure(bridgeInternalError,
                                              "Reflected constant slot \(index) is outside the argument buffer for kernel \(name)")
                            }
                            let offset = destinationAddress-bufferAddress
                            guard offset <= encodedLength &&
                                    UInt(data.count) <= encodedLength-offset else {
                                throw failure(bridgeInternalError,
                                              "Reflected constant slot \(index) is outside the argument buffer for kernel \(name)")
                            }
                            _ = data.withUnsafeBytes { bytes in
                                memcpy(destination, bytes.baseAddress!, bytes.count)
                            }
                        }
                    }
                    encoder.setBuffer(argumentBuffer, offset: 0, index: argumentBufferIndex)
                }

                encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: blockSize, height: 1, depth: 1))
                encoder.endEncoding()
                try queue.submitLocked(command, blocking: false,
                                       retainedResources: resources)
            }
        }
    }
}

private final class EventBox: @unchecked Sendable {
    let queue: QueueBox
    let event: any MTLSharedEvent
    private let stateLock = NSLock()
    private var signalCommand: (any MTLCommandBuffer)?
    private var value: UInt64 = 0

    init(queue: QueueBox) throws {
        guard let event = queue.device.makeSharedEvent() else {
            throw failure(bridgeUnsupported,
                          "The selected Metal device could not create a shared event")
        }
        self.queue = queue
        self.event = event
        event.label = "OpenMM Metal event"
    }

    func enqueue() throws {
        try stateLock.withBridgeLock {
            guard value != UInt64.max else {
                throw failure(bridgeOutOfRange, "Metal event signal value overflow")
            }
            try queue.checkForErrors()
            try queue.submissionLock.withBridgeLock {
                let command = try queue.makeCommandBuffer("signal event")
                let next = value + 1
                command.encodeSignalEvent(event, value: next)
                try queue.submitLocked(command, blocking: false,
                                       retainedResources: [event as AnyObject])
                value = next
                signalCommand = command
            }
        }
    }

    func wait() throws {
        let command: any MTLCommandBuffer = try stateLock.withBridgeLock {
            guard let signalCommand else {
                throw failure(bridgeInvalidArgument,
                              "Cannot wait for a Metal event before enqueue() has been called")
            }
            return signalCommand
        }
        command.waitUntilCompleted()
        guard command.status == .completed else {
            let detail = command.error.map(errorDescription) ?? "unknown Metal error"
            throw failure(bridgeNativeError, "Metal event signal failed: \(detail)")
        }
    }

    func queueWait(_ destination: QueueBox) throws {
        try stateLock.withBridgeLock {
            guard signalCommand != nil else {
                throw failure(bridgeInvalidArgument,
                              "Cannot enqueue a wait for a Metal event before enqueue() has been called")
            }
            guard destination.device.registryID == queue.device.registryID else {
                throw failure(bridgeInvalidArgument,
                              "Metal event cannot synchronize queues from different devices")
            }
            try destination.checkForErrors()
            try destination.submissionLock.withBridgeLock {
                let command = try destination.makeCommandBuffer("wait for event")
                command.encodeWaitForEvent(event, value: value)
                try destination.submitLocked(command, blocking: false,
                                             retainedResources: [event as AnyObject])
            }
        }
    }
}

// MARK: - C ABI

@c(OMMMetalErrorClear)
func bridgeErrorClear(_ error: UnsafeMutablePointer<OMMMetalErrorC>?) {
    guard let error else { return }
    if let message = error.pointee.message {
        free(message)
    }
    error.pointee.code = bridgeSuccess
    error.pointee.message = nil
}

@c(OMMMetalDeviceCapsClear)
func bridgeDeviceCapsClear(_ caps: UnsafeMutablePointer<OMMMetalDeviceCapsC>?) {
    guard let caps else { return }
    if let name = caps.pointee.name {
        free(name)
    }
    zeroCaps(caps)
}

@c(OMMMetalDeviceCount)
func bridgeDeviceCount(_ count: UnsafeMutablePointer<UInt>?,
                       _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let count else {
            throw failure(bridgeInvalidArgument, "Metal device count output is null")
        }
        count.pointee = UInt(MTLCopyAllDevices().count)
    }
}

@c(OMMMetalDeviceCapsAt)
func bridgeDeviceCapsAt(_ index: UInt,
                        _ caps: UnsafeMutablePointer<OMMMetalDeviceCapsC>?,
                        _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let caps else {
            throw failure(bridgeInvalidArgument, "Metal device capability output is null")
        }
        zeroCaps(caps)
        let devices = MTLCopyAllDevices()
        let nativeIndex = try checkedInt(index, "Metal device index")
        guard nativeIndex < devices.count else {
            throw failure(bridgeOutOfRange,
                          "Metal device index \(index) is out of range (\(devices.count) devices available)")
        }
        try writeCaps(DeviceSnapshot(device: devices[nativeIndex]), to: caps)
    }
}

@c(OMMMetalDefaultDevice)
func bridgeDefaultDevice(_ supported: UnsafeMutablePointer<UInt8>?,
                         _ caps: UnsafeMutablePointer<OMMMetalDeviceCapsC>?,
                         _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let supported, let caps else {
            throw failure(bridgeInvalidArgument, "Metal default device output is null")
        }
        supported.pointee = 0
        zeroCaps(caps)
        let device = MTLCreateSystemDefaultDevice()
        guard isSupportedDefaultDevice(device), let device else {
            return
        }
        try writeCaps(DeviceSnapshot(device: device), to: caps)
        supported.pointee = 1
    }
}

@c(OMMMetalQueueCreate)
func bridgeQueueCreate(_ deviceIndex: UInt,
                       _ output: UnsafeMutablePointer<OMMMetalQueueRef?>?,
                       _ caps: UnsafeMutablePointer<OMMMetalDeviceCapsC>?,
                       _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output, let caps else {
            throw failure(bridgeInvalidArgument, "Metal queue output is null")
        }
        output.pointee = nil
        zeroCaps(caps)
        let index = try checkedInt(deviceIndex, "Metal device index")
        let queue = try QueueBox(deviceIndex: index)
        try writeCaps(queue.caps, to: caps)
        output.pointee = Unmanaged.passRetained(queue).toOpaque()
    }
}

@c(OMMMetalQueueCreateSibling)
func bridgeQueueCreateSibling(_ parentHandle: OMMMetalQueueRef?,
                              _ output: UnsafeMutablePointer<OMMMetalQueueRef?>?,
                              _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument, "Metal sibling queue output is null")
        }
        output.pointee = nil
        let parent = try bridgeObject(parentHandle, as: QueueBox.self, "Metal parent queue")
        let sibling = try QueueBox(siblingOf: parent)
        output.pointee = Unmanaged.passRetained(sibling).toOpaque()
    }
}

@c(OMMMetalQueueRelease)
func bridgeQueueRelease(_ handle: OMMMetalQueueRef?) {
    guard let handle else { return }
    Unmanaged<QueueBox>.fromOpaque(handle).release()
}

@c(OMMMetalQueueWaitUntilIdle)
func bridgeQueueWaitUntilIdle(_ handle: OMMMetalQueueRef?,
                              _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let queue = try bridgeObject(handle, as: QueueBox.self, "Metal queue")
        try queue.waitUntilIdle()
    }
}

@c(OMMMetalQueueCheckForErrors)
func bridgeQueueCheckForErrors(_ handle: OMMMetalQueueRef?,
                               _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let queue = try bridgeObject(handle, as: QueueBox.self, "Metal queue")
        try queue.checkForErrors()
    }
}

@c(OMMMetalBufferCreate)
func bridgeBufferCreate(_ queueHandle: OMMMetalQueueRef?, _ byteCount: UInt,
                        _ name: UnsafePointer<CChar>?, _ nameLength: UInt,
                        _ output: UnsafeMutablePointer<OMMMetalBufferRef?>?,
                        _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument, "Metal buffer output is null")
        }
        output.pointee = nil
        let queue = try bridgeObject(queueHandle, as: QueueBox.self, "Metal queue")
        let bytes = try checkedInt(byteCount, "Metal buffer length")
        let labelLength = try checkedInt(nameLength, "Metal buffer name length")
        let label = try decodeUTF8(name, labelLength, "Metal buffer name")
        try queue.checkForErrors()
        let buffer = try BufferBox(queue: queue, bytes: bytes, name: label)
        output.pointee = Unmanaged.passRetained(buffer).toOpaque()
    }
}

@c(OMMMetalBufferRelease)
func bridgeBufferRelease(_ handle: OMMMetalBufferRef?) {
    guard let handle else { return }
    Unmanaged<BufferBox>.fromOpaque(handle).release()
}

@c(OMMMetalBufferResize)
func bridgeBufferResize(_ handle: OMMMetalBufferRef?, _ byteCount: UInt,
                        _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let buffer = try bridgeObject(handle, as: BufferBox.self, "Metal buffer")
        try buffer.resize(checkedInt(byteCount, "Metal buffer length"))
    }
}

@c(OMMMetalBufferUpload)
func bridgeBufferUpload(_ handle: OMMMetalBufferRef?, _ byteOffset: UInt,
                        _ source: UnsafeRawPointer?, _ byteCount: UInt, _ blocking: UInt8,
                        _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let buffer = try bridgeObject(handle, as: BufferBox.self, "Metal buffer")
        let offset = try checkedInt(byteOffset, "Metal upload offset")
        let bytes = try checkedInt(byteCount, "Metal upload length")
        try buffer.validate(offset: offset, bytes: bytes, operation: "Metal buffer upload")
        if bytes == 0 { return }
        guard let source else {
            throw failure(bridgeInvalidArgument, "Metal upload source is null")
        }
        try buffer.queue.checkForErrors()
        guard let staging = buffer.queue.device.makeBuffer(length: bytes,
                                                           options: .storageModeShared) else {
            throw failure(bridgeNativeError,
                          "Error allocating Metal staging buffer for uploading \(buffer.name) (\(bytes) bytes)")
        }
        memcpy(staging.contents(), source, bytes)

        try buffer.queue.submissionLock.withBridgeLock {
            let command = try buffer.queue.makeCommandBuffer("upload \(buffer.name)")
            guard let encoder = command.makeBlitCommandEncoder() else {
                throw failure(bridgeNativeError,
                              "Metal failed to create a blit encoder for uploading \(buffer.name)")
            }
            let destination = buffer.snapshot()
            encoder.copy(from: staging, sourceOffset: 0, to: destination,
                         destinationOffset: offset, size: bytes)
            encoder.endEncoding()
            try buffer.queue.submitLocked(command, blocking: blocking != 0,
                                          retainedResources: [staging as AnyObject,
                                                              destination as AnyObject])
        }
    }
}

@c(OMMMetalBufferDownload)
func bridgeBufferDownload(_ handle: OMMMetalBufferRef?, _ byteOffset: UInt,
                          _ destination: UnsafeMutableRawPointer?, _ byteCount: UInt,
                          _ blocking: UInt8,
                          _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let buffer = try bridgeObject(handle, as: BufferBox.self, "Metal buffer")
        let offset = try checkedInt(byteOffset, "Metal download offset")
        let bytes = try checkedInt(byteCount, "Metal download length")
        try buffer.validate(offset: offset, bytes: bytes, operation: "Metal buffer download")
        if bytes == 0 { return }
        guard let destination else {
            throw failure(bridgeInvalidArgument, "Metal download destination is null")
        }
        try buffer.queue.checkForErrors()
        guard let staging = buffer.queue.device.makeBuffer(length: bytes,
                                                           options: .storageModeShared) else {
            throw failure(bridgeNativeError,
                          "Error allocating Metal staging buffer for downloading \(buffer.name) (\(bytes) bytes)")
        }
        let request = DownloadRequest(destination: destination, staging: staging,
                                      byteCount: bytes)

        try buffer.queue.submissionLock.withBridgeLock {
            let command = try buffer.queue.makeCommandBuffer("download \(buffer.name)")
            guard let encoder = command.makeBlitCommandEncoder() else {
                throw failure(bridgeNativeError,
                              "Metal failed to create a blit encoder for downloading \(buffer.name)")
            }
            let source = buffer.snapshot()
            encoder.copy(from: source, sourceOffset: offset, to: staging,
                         destinationOffset: 0, size: bytes)
            encoder.endEncoding()
            try buffer.queue.submitLocked(command, blocking: blocking != 0,
                                          retainedResources: [source as AnyObject,
                                                              staging as AnyObject]) {
                request.complete($0)
            }
        }
    }
}

@c(OMMMetalBufferCopy)
func bridgeBufferCopy(_ sourceHandle: OMMMetalBufferRef?, _ sourceOffset: UInt,
                      _ destinationHandle: OMMMetalBufferRef?, _ destinationOffset: UInt,
                      _ byteCount: UInt,
                      _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let source = try bridgeObject(sourceHandle, as: BufferBox.self, "Metal source buffer")
        let destination = try bridgeObject(destinationHandle, as: BufferBox.self,
                                           "Metal destination buffer")
        guard source.queue === destination.queue else {
            throw failure(bridgeInvalidArgument,
                          "Copying between different Metal command queues is not supported")
        }
        let sourceByteOffset = try checkedInt(sourceOffset, "Metal source copy offset")
        let destinationByteOffset = try checkedInt(destinationOffset,
                                                   "Metal destination copy offset")
        let bytes = try checkedInt(byteCount, "Metal copy length")
        try source.validate(offset: sourceByteOffset, bytes: bytes,
                            operation: "Metal source copy")
        try destination.validate(offset: destinationByteOffset, bytes: bytes,
                                 operation: "Metal destination copy")
        if bytes == 0 { return }
        try source.queue.checkForErrors()
        try source.queue.submissionLock.withBridgeLock {
            let command = try source.queue.makeCommandBuffer("copy \(source.name) to \(destination.name)")
            guard let encoder = command.makeBlitCommandEncoder() else {
                throw failure(bridgeNativeError,
                              "Metal failed to create a blit encoder for buffer copy")
            }
            let nativeSource = source.snapshot()
            let nativeDestination = destination.snapshot()
            encoder.copy(from: nativeSource, sourceOffset: sourceByteOffset,
                         to: nativeDestination, destinationOffset: destinationByteOffset,
                         size: bytes)
            encoder.endEncoding()
            try source.queue.submitLocked(command, blocking: false,
                                          retainedResources: [nativeSource as AnyObject,
                                                              nativeDestination as AnyObject])
        }
    }
}

@c(OMMMetalProgramCreateSource)
func bridgeProgramCreateSource(_ queueHandle: OMMMetalQueueRef?,
                               _ source: UnsafePointer<CChar>?, _ sourceLength: UInt,
                               _ fastMath: UInt8, _ languageMajor: UInt16,
                               _ languageMinor: UInt16,
                               _ output: UnsafeMutablePointer<OMMMetalProgramRef?>?,
                               _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument, "Metal program output is null")
        }
        output.pointee = nil
        let queue = try bridgeObject(queueHandle, as: QueueBox.self, "Metal queue")
        let length = try checkedInt(sourceLength, "Metal program source length")
        let text = try decodeUTF8(source, length, "Metal program source", allowEmpty: false)
        try queue.checkForErrors()
        let program = try ProgramBox(queue: queue, source: text, fastMath: fastMath != 0,
                                     languageMajor: languageMajor,
                                     languageMinor: languageMinor)
        output.pointee = Unmanaged.passRetained(program).toOpaque()
    }
}

@c(OMMMetalProgramCreateLibrary)
func bridgeProgramCreateLibrary(_ queueHandle: OMMMetalQueueRef?,
                                _ libraryBytes: UnsafeRawPointer?, _ libraryLength: UInt,
                                _ output: UnsafeMutablePointer<OMMMetalProgramRef?>?,
                                _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument, "Metal program output is null")
        }
        output.pointee = nil
        let queue = try bridgeObject(queueHandle, as: QueueBox.self, "Metal queue")
        let length = try checkedInt(libraryLength, "Metal library length")
        guard length > 0, let libraryBytes else {
            throw failure(bridgeInvalidArgument, "Cannot load an empty Metal library")
        }
        try queue.checkForErrors()
        let program = try ProgramBox(queue: queue, libraryBytes: libraryBytes,
                                     length: length)
        output.pointee = Unmanaged.passRetained(program).toOpaque()
    }
}

@c(OMMMetalProgramRelease)
func bridgeProgramRelease(_ handle: OMMMetalProgramRef?) {
    guard let handle else { return }
    Unmanaged<ProgramBox>.fromOpaque(handle).release()
}

@c(OMMMetalKernelCreate)
func bridgeKernelCreate(_ programHandle: OMMMetalProgramRef?,
                        _ name: UnsafePointer<CChar>?, _ nameLength: UInt,
                        _ bindingMode: Int32, _ argumentBufferIndex: Int32,
                        _ output: UnsafeMutablePointer<OMMMetalKernelRef?>?,
                        _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument, "Metal kernel output is null")
        }
        output.pointee = nil
        let program = try bridgeObject(programHandle, as: ProgramBox.self, "Metal program")
        let length = try checkedInt(nameLength, "Metal kernel name length")
        let kernelName = try decodeUTF8(name, length, "Metal kernel name", allowEmpty: false)
        let kernel = try KernelBox(program: program, name: kernelName,
                                   bindingMode: bindingMode,
                                   argumentBufferIndex: Int(argumentBufferIndex))
        output.pointee = Unmanaged.passRetained(kernel).toOpaque()
    }
}

@c(OMMMetalKernelRelease)
func bridgeKernelRelease(_ handle: OMMMetalKernelRef?) {
    guard let handle else { return }
    Unmanaged<KernelBox>.fromOpaque(handle).release()
}

@c(OMMMetalKernelMaxBlockSize)
func bridgeKernelMaxBlockSize(_ handle: OMMMetalKernelRef?,
                              _ output: UnsafeMutablePointer<Int32>?,
                              _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument, "Metal kernel block-size output is null")
        }
        let kernel = try bridgeObject(handle, as: KernelBox.self, "Metal kernel")
        guard kernel.maxBlockSize() <= Int(Int32.max) else {
            throw failure(bridgeOutOfRange, "Metal kernel block size exceeds the C ABI range")
        }
        output.pointee = Int32(kernel.maxBlockSize())
    }
}

@c(OMMMetalKernelAddBuffer)
func bridgeKernelAddBuffer(_ kernelHandle: OMMMetalKernelRef?,
                           _ bufferHandle: OMMMetalBufferRef?,
                           _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let kernel = try bridgeObject(kernelHandle, as: KernelBox.self, "Metal kernel")
        let buffer = try bridgeObject(bufferHandle, as: BufferBox.self, "Metal buffer")
        try kernel.addBuffer(buffer)
    }
}

@c(OMMMetalKernelAddBytes)
func bridgeKernelAddBytes(_ kernelHandle: OMMMetalKernelRef?,
                          _ value: UnsafeRawPointer?, _ size: UInt,
                          _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let kernel = try bridgeObject(kernelHandle, as: KernelBox.self, "Metal kernel")
        let length = try checkedInt(size, "Metal primitive argument length")
        guard length > 0, let value else {
            throw failure(bridgeInvalidArgument,
                          "Primitive Metal kernel arguments must contain at least one byte")
        }
        try kernel.addBytes(value, size: length)
    }
}

@c(OMMMetalKernelAddEmpty)
func bridgeKernelAddEmpty(_ kernelHandle: OMMMetalKernelRef?,
                          _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let kernel = try bridgeObject(kernelHandle, as: KernelBox.self, "Metal kernel")
        kernel.addEmpty()
    }
}

@c(OMMMetalKernelSetBuffer)
func bridgeKernelSetBuffer(_ kernelHandle: OMMMetalKernelRef?, _ index: Int32,
                           _ bufferHandle: OMMMetalBufferRef?,
                           _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let kernel = try bridgeObject(kernelHandle, as: KernelBox.self, "Metal kernel")
        let buffer = try bridgeObject(bufferHandle, as: BufferBox.self, "Metal buffer")
        try kernel.setBuffer(index: Int(index), buffer: buffer)
    }
}

@c(OMMMetalKernelSetBytes)
func bridgeKernelSetBytes(_ kernelHandle: OMMMetalKernelRef?, _ index: Int32,
                          _ value: UnsafeRawPointer?, _ size: UInt,
                          _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let kernel = try bridgeObject(kernelHandle, as: KernelBox.self, "Metal kernel")
        let length = try checkedInt(size, "Metal primitive argument length")
        guard length > 0, let value else {
            throw failure(bridgeInvalidArgument,
                          "Primitive Metal kernel arguments must contain at least one byte")
        }
        try kernel.setBytes(index: Int(index), value: value, size: length)
    }
}

@c(OMMMetalKernelExecute)
func bridgeKernelExecute(_ handle: OMMMetalKernelRef?, _ threads: Int32,
                         _ blockSize: Int32,
                         _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let kernel = try bridgeObject(handle, as: KernelBox.self, "Metal kernel")
        try kernel.execute(threads: Int(threads), requestedBlockSize: Int(blockSize))
    }
}

@c(OMMMetalEventCreate)
func bridgeEventCreate(_ queueHandle: OMMMetalQueueRef?,
                       _ output: UnsafeMutablePointer<OMMMetalEventRef?>?,
                       _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument, "Metal event output is null")
        }
        output.pointee = nil
        let queue = try bridgeObject(queueHandle, as: QueueBox.self, "Metal queue")
        let event = try EventBox(queue: queue)
        output.pointee = Unmanaged.passRetained(event).toOpaque()
    }
}

@c(OMMMetalEventRelease)
func bridgeEventRelease(_ handle: OMMMetalEventRef?) {
    guard let handle else { return }
    Unmanaged<EventBox>.fromOpaque(handle).release()
}

@c(OMMMetalEventEnqueue)
func bridgeEventEnqueue(_ handle: OMMMetalEventRef?,
                        _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let event = try bridgeObject(handle, as: EventBox.self, "Metal event")
        try event.enqueue()
    }
}

@c(OMMMetalEventWait)
func bridgeEventWait(_ handle: OMMMetalEventRef?,
                     _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let event = try bridgeObject(handle, as: EventBox.self, "Metal event")
        try event.wait()
    }
}

@c(OMMMetalEventQueueWait)
func bridgeEventQueueWait(_ eventHandle: OMMMetalEventRef?,
                          _ queueHandle: OMMMetalQueueRef?,
                          _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        let event = try bridgeObject(eventHandle, as: EventBox.self, "Metal event")
        let queue = try bridgeObject(queueHandle, as: QueueBox.self, "Metal queue")
        try event.queueWait(queue)
    }
}
