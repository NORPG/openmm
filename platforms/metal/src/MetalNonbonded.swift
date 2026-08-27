/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * Swift-owned execution plan for the first native Metal NonbondedForce       *
 * implementation.  C++ supplies value snapshots through MetalSwiftBridge.h; *
 * topology validation, packing, GPU storage, and execution live here.        *
 * -------------------------------------------------------------------------- */

@preconcurrency import Foundation

/// A canonical, value-semantic particle pair.  Reversing the endpoints does
/// not create a distinct exception.
struct ParticlePair: Hashable, Comparable, Sendable {
    let first: UInt32
    let second: UInt32

    init(_ particle1: UInt32, _ particle2: UInt32) throws {
        guard particle1 != particle2 else {
            throw failure(bridgeInvalidArgument,
                          "A NonbondedForce exception cannot connect particle \(particle1) to itself")
        }
        first = min(particle1, particle2)
        second = max(particle1, particle2)
    }

    static func < (lhs: ParticlePair, rhs: ParticlePair) -> Bool {
        lhs.first == rhs.first ? lhs.second < rhs.second : lhs.first < rhs.first
    }
}

/// Immutable compressed-sparse-row exception adjacency.  Every exception has
/// one entry at each endpoint; the second lane identifies its parameter row.
struct NonbondedTopology: Equatable, Sendable {
    let exceptionPairs: [ParticlePair]
    let offsets: [UInt32]
    let entries: [SIMD2<UInt32>]

    init(particleCount: Int, exceptionPairs: [ParticlePair]) throws {
        guard particleCount >= 0 && UInt64(particleCount) <= UInt64(UInt32.max) else {
            throw failure(bridgeOutOfRange,
                          "The NonbondedForce particle count exceeds the Metal UInt32 range")
        }
        guard exceptionPairs.count <= Int(UInt32.max/2) else {
            throw failure(bridgeOutOfRange,
                          "The NonbondedForce exception adjacency exceeds the Metal UInt32 range")
        }

        var uniquePairs = Set<ParticlePair>()
        var adjacency = [[SIMD2<UInt32>]](repeating: [], count: particleCount)
        for (index, pair) in exceptionPairs.enumerated() {
            guard Int(pair.second) < particleCount else {
                throw failure(bridgeOutOfRange,
                              "NonbondedForce exception \(index) references particle \(pair.second), but only \(particleCount) particles exist")
            }
            guard uniquePairs.insert(pair).inserted else {
                throw failure(bridgeInvalidArgument,
                              "NonbondedForce contains duplicate exception pair (\(pair.first), \(pair.second))")
            }
            let parameterIndex = UInt32(index)
            adjacency[Int(pair.first)].append(SIMD2<UInt32>(pair.second, parameterIndex))
            adjacency[Int(pair.second)].append(SIMD2<UInt32>(pair.first, parameterIndex))
        }

        var builtOffsets = [UInt32](repeating: 0, count: particleCount+1)
        var builtEntries = [SIMD2<UInt32>]()
        builtEntries.reserveCapacity(exceptionPairs.count*2)
        for particle in 0..<particleCount {
            adjacency[particle].sort { lhs, rhs in lhs.x < rhs.x }
            builtOffsets[particle] = UInt32(builtEntries.count)
            builtEntries.append(contentsOf: adjacency[particle])
        }
        builtOffsets[particleCount] = UInt32(builtEntries.count)

        self.exceptionPairs = exceptionPairs
        offsets = builtOffsets
        entries = builtEntries
    }
}

/// A statically typed view of a private Metal buffer.  Constraining Element to
/// BitwiseCopyable makes host/device transfers explicit and prevents reference
/// carrying Swift values from accidentally crossing the GPU ABI.
struct TypedMetalBuffer<Element: BitwiseCopyable & Sendable>: Sendable {
    let raw: BufferBox
    let count: Int

    init(queue: QueueBox, count: Int, name: String) throws {
        guard count >= 0 else {
            throw failure(bridgeInvalidArgument,
                          "Cannot allocate a negative element count for \(name)")
        }
        let (bytes, overflow) = count.multipliedReportingOverflow(
            by: MemoryLayout<Element>.stride)
        guard !overflow else {
            throw failure(bridgeOutOfRange,
                          "The requested Metal buffer \(name) exceeds the native address range")
        }
        raw = try BufferBox(queue: queue, bytes: bytes, name: name)
        self.count = count
    }

    func upload(_ values: [Element], range: Range<Int>? = nil,
                blocking: Bool = true) throws {
        guard values.count == count else {
            throw failure(bridgeInvalidArgument,
                          "Metal upload expected \(count) values but received \(values.count)")
        }
        let selected = range ?? 0..<count
        try validate(selected, operation: "upload")
        if selected.isEmpty { return }

        let stride = MemoryLayout<Element>.stride
        try values.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else {
                throw failure(bridgeInternalError,
                              "A nonempty typed Metal upload has no storage")
            }
            try raw.upload(from: UnsafeRawPointer(base.advanced(by: selected.lowerBound)),
                           byteOffset: selected.lowerBound*stride,
                           byteCount: selected.count*stride,
                           blocking: blocking)
        }
    }

    func download(into values: inout [Element], blocking: Bool = true) throws {
        guard values.count == count else {
            throw failure(bridgeInvalidArgument,
                          "Metal download expected \(count) values but received \(values.count)")
        }
        if count == 0 { return }
        let bytes = count*MemoryLayout<Element>.stride
        try values.withUnsafeMutableBufferPointer { destination in
            guard let base = destination.baseAddress else {
                throw failure(bridgeInternalError,
                              "A nonempty typed Metal download has no storage")
            }
            try raw.download(to: UnsafeMutableRawPointer(base), byteOffset: 0,
                             byteCount: bytes, blocking: blocking)
        }
    }

    private func validate(_ range: Range<Int>, operation: String) throws {
        guard range.lowerBound >= 0 && range.upperBound <= count else {
            throw failure(bridgeOutOfRange,
                          "Typed Metal \(operation) range is outside 0..<\(count)")
        }
    }
}

/// Intent-revealing execution flags while retaining the two scalar arguments
/// used by the initial MSL kernel ABI.
struct NonbondedExecutionOptions: OptionSet, Sendable {
    let rawValue: UInt32

    static let forces = NonbondedExecutionOptions(rawValue: 1 << 0)
    static let energy = NonbondedExecutionOptions(rawValue: 1 << 1)
}

struct NonbondedModel: Sendable {
    let particleParameters: [SIMD4<Float>]
    let exceptionParameters: [SIMD4<Float>]
    let topology: NonbondedTopology

    var particleCount: Int { particleParameters.count }
    var exceptionCount: Int { exceptionParameters.count }

    init(particles: UnsafePointer<OMMMetalNonbondedParticleC>?, particleCount: Int,
         exceptions: UnsafePointer<OMMMetalNonbondedExceptionC>?, exceptionCount: Int) throws {
        guard particleCount >= 0 && UInt64(particleCount) <= UInt64(UInt32.max) else {
            throw failure(bridgeOutOfRange,
                          "The NonbondedForce particle count exceeds the Metal UInt32 range")
        }
        guard exceptionCount >= 0 && UInt64(exceptionCount) <= UInt64(UInt32.max) else {
            throw failure(bridgeOutOfRange,
                          "The NonbondedForce exception count exceeds the Metal UInt32 range")
        }
        if particleCount != 0 && particles == nil {
            throw failure(bridgeInvalidArgument,
                          "NonbondedForce particle parameters are null for a nonempty System")
        }
        if exceptionCount != 0 && exceptions == nil {
            throw failure(bridgeInvalidArgument,
                          "NonbondedForce exception parameters are null for a nonempty exception list")
        }

        var packedParticles = [SIMD4<Float>]()
        packedParticles.reserveCapacity(particleCount)
        if let particles {
            for index in 0..<particleCount {
                let value = particles[index]
                packedParticles.append(try Self.pack(charge: value.charge,
                                                     sigma: value.sigma,
                                                     epsilon: value.epsilon,
                                                     description: "particle \(index)"))
            }
        }

        var pairs = [ParticlePair]()
        var packedExceptions = [SIMD4<Float>]()
        pairs.reserveCapacity(exceptionCount)
        packedExceptions.reserveCapacity(exceptionCount)
        if let exceptions {
            for index in 0..<exceptionCount {
                let value = exceptions[index]
                pairs.append(try ParticlePair(value.particle1, value.particle2))
                packedExceptions.append(try Self.pack(charge: value.chargeProd,
                                                       sigma: value.sigma,
                                                       epsilon: value.epsilon,
                                                       description: "exception \(index)"))
            }
        }

        particleParameters = packedParticles
        exceptionParameters = packedExceptions
        topology = try NonbondedTopology(particleCount: particleCount,
                                         exceptionPairs: pairs)
    }

    private static func pack(charge: Double, sigma: Double, epsilon: Double,
                             description: String) throws -> SIMD4<Float> {
        guard charge.isFinite && sigma.isFinite && epsilon.isFinite else {
            throw failure(bridgeInvalidArgument,
                          "NonbondedForce \(description) contains a non-finite parameter")
        }
        let packed = SIMD4<Float>(Float(charge), Float(sigma), Float(epsilon), 0)
        guard packed.x.isFinite && packed.y.isFinite && packed.z.isFinite else {
            throw failure(bridgeOutOfRange,
                          "NonbondedForce \(description) cannot be represented by the Metal Float parameter ABI")
        }
        return packed
    }
}

/// ARC-owned, synchronous execution plan exposed to C++ as an opaque handle.
/// The lock makes a parameter update one operation relative to kernel dispatch
/// and energy reduction, while the candidate model provides transactional
/// value semantics before any ranged upload is attempted.
final class NonbondedKernelPlan: @unchecked Sendable {
    private let transactionLock = NSLock()
    private let positionBuffer: BufferBox
    private let forceBuffer: BufferBox
    private let program: ProgramBox
    private let kernel: KernelBox
    private let particles: TypedMetalBuffer<SIMD4<Float>>
    private let exceptionOffsets: TypedMetalBuffer<UInt32>
    private let exceptionEntries: TypedMetalBuffer<SIMD2<UInt32>>
    private let exceptions: TypedMetalBuffer<SIMD4<Float>>
    private let energyByParticle: TypedMetalBuffer<Float>
    private var model: NonbondedModel

    init(positionBuffer: BufferBox, forceBuffer: BufferBox,
         libraryBytes: UnsafeRawPointer, libraryLength: Int,
         model: NonbondedModel) throws {
        guard positionBuffer.queue === forceBuffer.queue else {
            throw failure(bridgeInvalidArgument,
                          "NonbondedForce position and force buffers belong to different Metal queues")
        }
        let vectorBytes = try Self.checkedVectorBytes(model.particleCount)
        try positionBuffer.validate(offset: 0, bytes: vectorBytes,
                                    operation: "NonbondedForce position buffer")
        try forceBuffer.validate(offset: 0, bytes: vectorBytes,
                                 operation: "NonbondedForce force buffer")

        let queue = positionBuffer.queue
        let program = try ProgramBox(queue: queue, libraryBytes: libraryBytes,
                                     length: libraryLength)
        let kernel = try KernelBox(program: program,
                                   name: "computeNoCutoffNonbonded",
                                   bindingMode: 0, argumentBufferIndex: 0)
        let particles = try TypedMetalBuffer<SIMD4<Float>>(
            queue: queue, count: model.particleCount,
            name: "Metal nonbonded particle parameters")
        let exceptionOffsets = try TypedMetalBuffer<UInt32>(
            queue: queue, count: model.topology.offsets.count,
            name: "Metal nonbonded exception offsets")
        let exceptionEntries = try TypedMetalBuffer<SIMD2<UInt32>>(
            queue: queue, count: model.topology.entries.count,
            name: "Metal nonbonded exception entries")
        let exceptions = try TypedMetalBuffer<SIMD4<Float>>(
            queue: queue, count: model.exceptionCount,
            name: "Metal nonbonded exception parameters")
        let energyByParticle = try TypedMetalBuffer<Float>(
            queue: queue, count: model.particleCount,
            name: "Metal nonbonded energy")

        try particles.upload(model.particleParameters)
        try exceptionOffsets.upload(model.topology.offsets)
        try exceptionEntries.upload(model.topology.entries)
        try exceptions.upload(model.exceptionParameters)

        try kernel.addBuffer(positionBuffer)
        try kernel.addBuffer(forceBuffer)
        try kernel.addBuffer(particles.raw)
        try kernel.addBuffer(exceptionOffsets.raw)
        try kernel.addBuffer(exceptionEntries.raw)
        try kernel.addBuffer(exceptions.raw)
        try kernel.addBuffer(energyByParticle.raw)
        try Self.add(UInt32(model.particleCount), to: kernel)
        try Self.add(UInt32(0), to: kernel)
        try Self.add(UInt32(0), to: kernel)

        self.positionBuffer = positionBuffer
        self.forceBuffer = forceBuffer
        self.program = program
        self.kernel = kernel
        self.particles = particles
        self.exceptionOffsets = exceptionOffsets
        self.exceptionEntries = exceptionEntries
        self.exceptions = exceptions
        self.energyByParticle = energyByParticle
        self.model = model
    }

    func execute(options: NonbondedExecutionOptions) throws -> Double {
        try transactionLock.withBridgeLock {
            guard !options.isEmpty && model.particleCount != 0 else { return 0.0 }

            let forceFlag: UInt32 = options.contains(.forces) ? 1 : 0
            let energyFlag: UInt32 = options.contains(.energy) ? 1 : 0
            try Self.set(forceFlag, index: 8, on: kernel)
            try Self.set(energyFlag, index: 9, on: kernel)
            try kernel.execute(threads: model.particleCount, requestedBlockSize: -1)

            guard options.contains(.energy) else { return 0.0 }
            var values = [Float](repeating: 0, count: model.particleCount)
            try energyByParticle.download(into: &values)
            return values.reduce(into: 0.0) { total, value in
                total += Double(value)
            }
        }
    }

    func update(with candidate: NonbondedModel,
                firstParticle: Int32, lastParticle: Int32,
                firstException: Int32, lastException: Int32) throws {
        try transactionLock.withBridgeLock {
            guard candidate.particleCount == model.particleCount else {
                throw failure(bridgeInvalidArgument,
                              "updateParametersInContext: the number of nonbonded particles changed from \(model.particleCount) to \(candidate.particleCount)")
            }
            guard candidate.exceptionCount == model.exceptionCount else {
                throw failure(bridgeInvalidArgument,
                              "updateParametersInContext: the number of nonbonded exceptions changed from \(model.exceptionCount) to \(candidate.exceptionCount)")
            }
            guard candidate.topology.exceptionPairs == model.topology.exceptionPairs else {
                throw failure(bridgeInvalidArgument,
                              Self.topologyChangeDescription(old: model.topology.exceptionPairs,
                                                             new: candidate.topology.exceptionPairs))
            }

            let particleRange = try Self.changedRange(first: firstParticle,
                                                      last: lastParticle,
                                                      count: model.particleCount,
                                                      description: "particle")
            let exceptionRange = try Self.changedRange(first: firstException,
                                                       last: lastException,
                                                       count: model.exceptionCount,
                                                       description: "exception")
            if let particleRange {
                try particles.upload(candidate.particleParameters,
                                     range: particleRange, blocking: true)
            }
            if let exceptionRange {
                try exceptions.upload(candidate.exceptionParameters,
                                      range: exceptionRange, blocking: true)
            }
            model = candidate
        }
    }

    private static func checkedVectorBytes(_ count: Int) throws -> Int {
        let (bytes, overflow) = count.multipliedReportingOverflow(
            by: MemoryLayout<SIMD4<Float>>.stride)
        guard !overflow else {
            throw failure(bridgeOutOfRange,
                          "The NonbondedForce context buffer size exceeds the native address range")
        }
        return bytes
    }

    private static func changedRange(first: Int32, last: Int32, count: Int,
                                     description: String) throws -> Range<Int>? {
        if first > last { return nil }
        guard first >= 0 && last >= 0 && Int(last) < count else {
            throw failure(bridgeOutOfRange,
                          "updateParametersInContext: changed nonbonded \(description) range \(first)...\(last) is outside 0..<\(count)")
        }
        return Int(first)..<(Int(last)+1)
    }

    private static func topologyChangeDescription(old: [ParticlePair],
                                                  new: [ParticlePair]) -> String {
        if let index = old.indices.first(where: { old[$0] != new[$0] }) {
            return "updateParametersInContext: exception \(index) endpoints changed from (\(old[index].first), \(old[index].second)) to (\(new[index].first), \(new[index].second))"
        }
        return "updateParametersInContext: the NonbondedForce exception topology changed"
    }

    private static func add<T: BitwiseCopyable & Sendable>(_ value: T,
                                                            to kernel: KernelBox) throws {
        var copy = value
        try withUnsafePointer(to: &copy) { pointer in
            try kernel.addBytes(UnsafeRawPointer(pointer), size: MemoryLayout<T>.size)
        }
    }

    private static func set<T: BitwiseCopyable & Sendable>(_ value: T, index: Int,
                                                            on kernel: KernelBox) throws {
        var copy = value
        try withUnsafePointer(to: &copy) { pointer in
            try kernel.setBytes(index: index, value: UnsafeRawPointer(pointer),
                                size: MemoryLayout<T>.size)
        }
    }
}

// MARK: - Nonbonded C ABI

@c(OMMMetalNonbondedCreate)
func bridgeNonbondedCreate(
    _ positionHandle: OMMMetalBufferRef?, _ forceHandle: OMMMetalBufferRef?,
    _ libraryBytes: UnsafeRawPointer?, _ libraryLength: UInt,
    _ particles: UnsafePointer<OMMMetalNonbondedParticleC>?, _ particleCount: UInt,
    _ exceptions: UnsafePointer<OMMMetalNonbondedExceptionC>?, _ exceptionCount: UInt,
    _ output: UnsafeMutablePointer<OMMMetalNonbondedRef?>?,
    _ error: UnsafeMutablePointer<OMMMetalErrorC>?
) -> Int32 {
    bridgeCall(error) {
        guard let output else {
            throw failure(bridgeInvalidArgument,
                          "Metal nonbonded plan output is null")
        }
        output.pointee = nil
        let positionBuffer = try bridgeObject(positionHandle, as: BufferBox.self,
                                              "Metal position buffer")
        let forceBuffer = try bridgeObject(forceHandle, as: BufferBox.self,
                                           "Metal force buffer")
        let libraryLength = try checkedInt(libraryLength, "Metal nonbonded library length")
        guard libraryLength > 0, let libraryBytes else {
            throw failure(bridgeInvalidArgument,
                          "Cannot create a Metal nonbonded plan from an empty library")
        }
        let particleCount = try checkedInt(particleCount,
                                           "Metal nonbonded particle count")
        let exceptionCount = try checkedInt(exceptionCount,
                                            "Metal nonbonded exception count")
        let model = try NonbondedModel(particles: particles,
                                       particleCount: particleCount,
                                       exceptions: exceptions,
                                       exceptionCount: exceptionCount)
        let plan = try NonbondedKernelPlan(positionBuffer: positionBuffer,
                                           forceBuffer: forceBuffer,
                                           libraryBytes: libraryBytes,
                                           libraryLength: libraryLength,
                                           model: model)
        output.pointee = Unmanaged.passRetained(plan).toOpaque()
    }
}

@c(OMMMetalNonbondedRelease)
func bridgeNonbondedRelease(_ handle: OMMMetalNonbondedRef?) {
    guard let handle else { return }
    Unmanaged<NonbondedKernelPlan>.fromOpaque(handle).release()
}

@c(OMMMetalNonbondedExecute)
func bridgeNonbondedExecute(_ handle: OMMMetalNonbondedRef?,
                            _ includeForces: UInt8, _ includeEnergy: UInt8,
                            _ energy: UnsafeMutablePointer<Double>?,
                            _ error: UnsafeMutablePointer<OMMMetalErrorC>?) -> Int32 {
    bridgeCall(error) {
        guard let energy else {
            throw failure(bridgeInvalidArgument,
                          "Metal nonbonded energy output is null")
        }
        energy.pointee = 0
        let plan = try bridgeObject(handle, as: NonbondedKernelPlan.self,
                                    "Metal nonbonded plan")
        var options: NonbondedExecutionOptions = []
        if includeForces != 0 { options.insert(.forces) }
        if includeEnergy != 0 { options.insert(.energy) }
        energy.pointee = try plan.execute(options: options)
    }
}

@c(OMMMetalNonbondedUpdate)
func bridgeNonbondedUpdate(
    _ handle: OMMMetalNonbondedRef?,
    _ particles: UnsafePointer<OMMMetalNonbondedParticleC>?, _ particleCount: UInt,
    _ exceptions: UnsafePointer<OMMMetalNonbondedExceptionC>?, _ exceptionCount: UInt,
    _ firstParticle: Int32, _ lastParticle: Int32,
    _ firstException: Int32, _ lastException: Int32,
    _ error: UnsafeMutablePointer<OMMMetalErrorC>?
) -> Int32 {
    bridgeCall(error) {
        let plan = try bridgeObject(handle, as: NonbondedKernelPlan.self,
                                    "Metal nonbonded plan")
        let particleCount = try checkedInt(particleCount,
                                           "Metal nonbonded particle count")
        let exceptionCount = try checkedInt(exceptionCount,
                                            "Metal nonbonded exception count")
        let candidate = try NonbondedModel(particles: particles,
                                           particleCount: particleCount,
                                           exceptions: exceptions,
                                           exceptionCount: exceptionCount)
        try plan.update(with: candidate,
                        firstParticle: firstParticle, lastParticle: lastParticle,
                        firstException: firstException, lastException: lastException)
    }
}
