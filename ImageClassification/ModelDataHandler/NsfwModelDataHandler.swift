//
//  ModelDataHandler.swift
//  Runner
//
//  Created by dave on 2019/11/13.
//  Copyright © 2019 The Chromium Authors. All rights reserved.
//
import CoreImage
import TensorFlowLite
import UIKit

/// A result from invoking the `Interpreter`.
class NsfwResult {
    var nsfw: Float = 0.0
    var sfw: Float = 0.0
}
enum NsResult<T> {
    case success(T)
    case error(Error)
}

/// Define errors that could happen in when doing image clasification
enum ClassificationError: Error {
    // Invalid input image
    case invalidImage
    // TF Lite Internal Error when initializing
    case internalError(Error)
}
/// Information about a model file or labels file.
typealias NsfwFileInfo = (name: String, extension: String)
/// Information about the MobileNet model.
enum NsfwMobileNet {
    static let modelInfo: NsfwFileInfo = (name: "nsfw", extension: "tflite")
}
class NsfwModelDataHandler {
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int

    // MARK: - Model Parameters
    
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 224
    let inputHeight = 224
    
    
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter
    
    /// Information about the alpha component in RGBA data.
    private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
    private var inputImageWidth: Int
    private var inputImageHeight: Int
    // MARK: - Initialization
    
    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: NsfwFileInfo,threadCount: Int = 10) {
        let modelFilename = modelFileInfo.name
        
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
            ) else {
                print("Failed to load the model file with name: \(modelFilename).")
                return nil
        }
        
        // Specify the options for the `Interpreter`.
        self.threadCount = threadCount
        var options = InterpreterOptions()
        options.threadCount = threadCount
        do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
            
            // Read TF Lite model input dimension
            let inputShape = try interpreter.input(at: 0).shape
            self.inputImageWidth = inputShape.dimensions[1]
            self.inputImageHeight = inputShape.dimensions[2]
            print("inputImageWidth:\(inputImageWidth) inputImageHeight: \(inputImageHeight)")
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
       
    }

    // MARK: - Internal Methods

    /// Run image classification on the input image.
    ///
    /// - Parameters
    ///   - image: an UIImage instance to classify.
    ///   - completion: callback to receive the classification result.
    func classify(image: UIImage, completion: @escaping ((NsResult<NsfwResult>) -> ())) {
        DispatchQueue.global(qos: .background).async {
            let outputTensor: Tensor
            do {
                // Preprocessing: Convert the input UIImage to (256 x 256) grayscale image to feed to TF Lite model.
                guard let rgbData = image.scaledData(with: CGSize(width:224, height: 224))
                else {
                        DispatchQueue.main.async {
                            completion(.error(ClassificationError.invalidImage))
                        }
                        print("Failed to convert the image buffer to RGB data.")
                        return
                }
                var inputData = Data.init(capacity: 0)
                for row in 0...self.inputImageWidth-1 {
                    for col in 0...self.inputImageHeight-1 {
                       // print("row: \(row)  col:\(col) ")
                        let offset = 4 * (row * self.inputImageWidth  + col)
                       // print("offset: \(offset)")
                        let red = Float32(rgbData[offset])
                        //print("red: \(red)")
                        let green = Float32(rgbData[offset+1])
                        //print("green: \(green)")
                        let blue = Float32(rgbData[offset+2])
                        //print("blue: \(blue)")
                        var red1 =  blue - 123
                        var green1 = green - 117
                        var blue1 = red - 104
                        inputData.append(Data(buffer: UnsafeBufferPointer(start: &blue1, count: 1)))
                        inputData.append(Data(buffer: UnsafeBufferPointer(start: &green1, count: 1)))
                        inputData.append(Data(buffer: UnsafeBufferPointer(start: &red1, count: 1)))
                    }
                }
                // Allocate memory for the model's input `Tensor`s.
                try self.interpreter.allocateTensors()
                
                // Copy the RGB data to the input `Tensor`.
                //try self.interpreter.copy(rgbData, toInputAt: 0)
                try self.interpreter.copy(inputData, toInputAt: 0)
                
                // Run inference by invoking the `Interpreter`.
                try self.interpreter.invoke()
                
                // Get the output `Tensor` to process the inference results.
                outputTensor = try self.interpreter.output(at: 0)
            } catch let error {
                print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    completion(.error(ClassificationError.internalError(error)))
                }
                return
            }
            
            // Postprocessing: Find the label with highest confidence and return as human readable text.
            let results = outputTensor.data.toArray(type: Float32.self)
            //let results = outputTensor.data.toArray(type: Float32.self)
            print("results: \(results)")
//            let maxConfidence = results.max() ?? -1
//            let maxIndex = results.firstIndex(of: maxConfidence) ?? -1
//            let humanReadableResult = "Predicted: \(maxIndex)\nConfidence: \(maxConfidence)"
//
            
            // Return the classification result
            DispatchQueue.main.async {
                let n = NsfwResult.init()
                n.nsfw = results[1]
                n.sfw = results[0]
                completion(.success(n))
            }
        }
    }
    
    // MARK: - Private Methods
    
    /// Returns the top N inference results sorted in descending order.
//    private func getTopN(results: [Float]) -> [Inference] {
//        // Create a zipped array of tuples [(labelIndex: Int, confidence: Float)].
//        let zippedResults = zip(labels.indices, results)
//
//        // Sort the zipped results by confidence value in descending order.
//        let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)
//
//        // Return the `Inference` results.
//        return sortedResults.map { result in Inference(confidence: result.1, label: labels[result.0]) }
//    }
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
//    private func loadLabels(fileInfo: FileInfo) {
//        let filename = fileInfo.name
//        let fileExtension = fileInfo.extension
//        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
//            fatalError("Labels file not found in bundle. Please add a labels file with name " +
//                "\(filename).\(fileExtension) and try again.")
//        }
//        do {
//            let contents = try String(contentsOf: fileURL, encoding: .utf8)
//            labels = contents.components(separatedBy: .newlines)
//        } catch {
//            fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
//                "valid labels file and try again.")
//        }
//    }
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
//    private func rgbDataFromBuffer(
//        _ buffer: CVPixelBuffer,
//        byteCount: Int,
//        isModelQuantized: Bool
//        ) -> Data? {
//        CVPixelBufferLockBaseAddress(buffer, .readOnly)
//        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
//        guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
//            return nil
//        }
//        let count = CVPixelBufferGetDataSize(buffer)
//        let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
//        var rgbBytes = [UInt8](repeating: 0, count: byteCount)
//        var index = 0
//
//        let pixelBufferFormat = CVPixelBufferGetPixelFormatType(buffer)
//        var rgbChannelMap : [Int]
//        switch (pixelBufferFormat) {
//        case kCVPixelFormatType_32BGRA:
//            rgbChannelMap = [2, 1, 0]
//        case kCVPixelFormatType_32RGBA:
//            rgbChannelMap = [0, 1, 2]
//        case kCVPixelFormatType_32ABGR:
//            rgbChannelMap = [3, 2, 1]
//        case kCVPixelFormatType_32ARGB:
//            rgbChannelMap = [1, 2, 3]
//        default:
//            // Unknown pixel format.
//            return nil
//        }
//
//        // Iterate through pixels and reorder bytes to be in RGB order.
//        let numChannels = 4
//        for pixelIndex in 0..<count / numChannels {
//            let offset = pixelIndex * numChannels
//            for j in 0...2 {
//                rgbBytes[index] = bufferData[offset + rgbChannelMap[j]]
//                index += 1
//            }
//        }
//
//        if isModelQuantized { return Data(bytes: rgbBytes) }
//        return Data(copyingBufferOf: rgbBytes.map { Float($0) / 255.0 })
//    }
}

//extension NSMutableData {
//    mutating func append<T>(value: T) {
//        Swift.withUnsafeBytes(of: value) { buffer in
//            self.append(buffer.bindMemory(to: UInt8.self))
//        }
//    }
//}

extension Data {
  func toArray<T>(type: T.Type) -> [T] where T: ExpressibleByIntegerLiteral {
     var array = Array<T>(repeating: 0, count: self.count/MemoryLayout<T>.stride)
     _ = array.withUnsafeMutableBytes { copyBytes(to: $0) }
     return array
   }
}
