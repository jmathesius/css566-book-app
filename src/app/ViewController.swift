import UIKit
import AVFoundation
import Vision
import Photos

// MARK: - Codable Structs
// For handling the server's JSON response
struct UploadResponse: Codable {
    let source: String
    let data: ResponseData?
    let error: String?
}

struct ResponseData: Codable {
    let id: Int?
    let image_name: String?
}

// The core model object for a detected spine.
// It is made Codable to be serialized into JSON for upload.
struct OrientedBoundingBox: Codable {
    let center: CGPoint
    let size: CGSize
    let angle: CGFloat // in radians
    let confidence: Float
    
    // Custom coding keys are needed because native types like CGPoint are not directly Codable.
    enum CodingKeys: String, CodingKey {
        case centerX, centerY, width, height, angle, confidence
    }

    // Custom encoder to flatten the CGPoint and CGSize structures into the JSON.
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(center.x, forKey: .centerX)
        try container.encode(center.y, forKey: .centerY)
        try container.encode(size.width, forKey: .width)
        try container.encode(size.height, forKey: .height)
        try container.encode(angle, forKey: .angle)
        try container.encode(confidence, forKey: .confidence)
    }

    // Custom decoder to reconstruct the native types from the flattened JSON.
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let centerX = try container.decode(CGFloat.self, forKey: .centerX)
        let centerY = try container.decode(CGFloat.self, forKey: .centerY)
        self.center = CGPoint(x: centerX, y: centerY)
        
        let width = try container.decode(CGFloat.self, forKey: .width)
        let height = try container.decode(CGFloat.self, forKey: .height)
        self.size = CGSize(width: width, height: height)
        
        self.angle = try container.decode(CGFloat.self, forKey: .angle)
        self.confidence = try container.decode(Float.self, forKey: .confidence)
    }

    // Standard initializer used by the app's internal logic (e.g., model parsing).
    init(center: CGPoint, size: CGSize, angle: CGFloat, confidence: Float) {
        self.center = center
        self.size = size
        self.angle = angle
        self.confidence = confidence
    }
    
    // This computed property is a helper for drawing the box on screen and is not part of the encoded JSON.
    var corners: [CGPoint] {
        let w = self.size.width
        let h = self.size.height
        let points = [CGPoint(x: -w/2, y: -h/2), CGPoint(x: w/2, y: -h/2), CGPoint(x: w/2, y: h/2), CGPoint(x: -w/2, y: h/2)]
        let cosAngle = cos(self.angle)
        let sinAngle = sin(self.angle)
        return points.map { p in
            let rotatedX = p.x * cosAngle - p.y * sinAngle + self.center.x
            let rotatedY = p.x * sinAngle + p.y * cosAngle + self.center.y
            return CGPoint(x: rotatedX, y: rotatedY)
        }
    }
}


// MARK: - Network Service
// Manages all API communication for uploading images, crops, and metadata.
class NetworkService {
    
    static let shared = NetworkService()
    private init() {}
    
    // !!! IMPORTANT: Replace this with your actual server URL !!!
    private let baseURL = "https://smelly-rompish-jonah.ngrok-free.dev"
    private lazy var cropUploadURL = URL(string: "\(baseURL)/upload")!
    private lazy var wholeImageUploadURL = URL(string: "\(baseURL)/upload_whole")!

    // Generic private upload function that handles constructing the multipart form data.
    private func upload(
        image: UIImage,
        to endpoint: URL,
        with wholeImageID: String,
        boundingBoxesJSON: String?, // Now accepts an optional JSON string for bounding boxes
        completion: @escaping (Result<UploadResponse, Error>) -> Void
    ) {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        let imageData = image.jpegData(compressionQuality: 0.8)!
        var body = Data()
        
        // Field 1: whole_image_id (sent with every request)
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"whole_image_id\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(wholeImageID)\r\n".data(using: .utf8)!)
        
        // Field 2: bounding_boxes_json (sent ONLY with the whole image upload)
        if let json = boundingBoxesJSON {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"bounding_boxes_json\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(json)\r\n".data(using: .utf8)!)
        }
        
        // Field 3: The image file data
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error { completion(.failure(error)); return }
            guard let data = data else { completion(.failure(URLError(.cannotParseResponse))); return }
            do {
                let decodedResponse = try JSONDecoder().decode(UploadResponse.self, from: data)
                completion(.success(decodedResponse))
            } catch {
                print("JSON Decoding Error: \(error)")
                if let responseString = String(data: data, encoding: .utf8) { print("Server Response: \(responseString)") }
                completion(.failure(error))
            }
        }
        task.resume()
    }
    
    // Public convenience method for uploading a single crop.
    func uploadCrop(image: UIImage, wholeImageID: String, completion: @escaping (Result<UploadResponse, Error>) -> Void) {
        upload(image: image, to: cropUploadURL, with: wholeImageID, boundingBoxesJSON: nil, completion: completion)
    }
    
    // Public convenience method for uploading the main image along with its detected bounding boxes.
    func uploadWholeImage(image: UIImage, wholeImageID: String, boundingBoxesJSON: String, completion: @escaping (Result<UploadResponse, Error>) -> Void) {
        upload(image: image, to: wholeImageUploadURL, with: wholeImageID, boundingBoxesJSON: boundingBoxesJSON, completion: completion)
    }
}


// MARK: - ViewController
class ViewController: UIViewController {

    // MARK: - UI & AV Foundation Properties
    private let captureSession = AVCaptureSession()
    private lazy var previewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    private let videoDataOutput = AVCaptureVideoDataOutput()
    private var boundingBoxLayer = CALayer()

    // MARK: - Vision & Model Properties
    private var visionRequests = [VNRequest]()
    private let modelInputSize = CGSize(width: 640, height: 640)
    private var bufferSize: CGSize = .zero
    private let confidenceThreshold: Float = 0.7
    private let iouThreshold: Float = 0.4
    private var latestDetections: [OrientedBoundingBox] = []
    private var latestPixelBuffer: CVPixelBuffer?
    private let visionQueue = DispatchQueue(label: "visionQueue")
    
    // MARK: - Data Collection Properties & UI
    private var imagesTakenCount = 0
    private var cropsUploadedCount = 0
    
    private lazy var statsLabel: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.6)
        label.textAlignment = .center
        label.font = .monospacedSystemFont(ofSize: 16, weight: .semibold)
        label.layer.cornerRadius = 8
        label.clipsToBounds = true
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private lazy var captureButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Capture Spines", for: .normal)
        button.backgroundColor = UIColor.systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 8
        button.translatesAutoresizingMaskIntoConstraints = false
        button.addTarget(self, action: #selector(captureButtonTapped), for: .touchUpInside)
        return button
    }()
    
    private lazy var activityIndicator: UIActivityIndicatorView = {
        let indicator = UIActivityIndicatorView(style: .large)
        indicator.color = .white
        indicator.hidesWhenStopped = true
        indicator.translatesAutoresizingMaskIntoConstraints = false
        return indicator
    }()
    
    // MARK: - Lifecycle Methods
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupDrawingLayers()
        setupVision()
        setupUI()
        updateStatsLabel()
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.previewLayer.frame = self.view.bounds
        self.boundingBoxLayer.frame = self.view.bounds
    }

    // MARK: - Setup Methods
    private func setupCamera() {
        guard let camera = AVCaptureDevice.default(for: .video) else {
            print("Error: No camera available.")
            return
        }
        
        do {
            let cameraInput = try AVCaptureDeviceInput(device: camera)
            captureSession.addInput(cameraInput)
        } catch {
            print("Error setting up camera input: \(error)")
            return
        }
        
        videoDataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoDataOutput)
        previewLayer.videoGravity = .resizeAspectFill
        self.view.layer.addSublayer(previewLayer)
    }
    
    private func setupDrawingLayers() {
        boundingBoxLayer.frame = self.view.bounds
        self.view.layer.addSublayer(boundingBoxLayer)
    }
    
    private func setupUI() {
        view.addSubview(statsLabel)
        view.addSubview(captureButton)
        view.addSubview(activityIndicator)
        
        NSLayoutConstraint.activate([
            statsLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            statsLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            statsLabel.widthAnchor.constraint(equalToConstant: 300),
            statsLabel.heightAnchor.constraint(equalToConstant: 40),
            
            captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            captureButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            captureButton.widthAnchor.constraint(equalToConstant: 200),
            captureButton.heightAnchor.constraint(equalToConstant: 50),
            
            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])
    }
    
    private func setupVision() {
        do {
            // Replace `epoch_79` with the actual name of your Core ML model class.
            let configuration = MLModelConfiguration()
            let coreMLModel = try epoch_79(configuration: configuration)
            let model = try VNCoreMLModel(for: coreMLModel.model)
            
            let request = VNCoreMLRequest(model: model) { [weak self] request, error in
                self?.handleVisionRequest(request: request, error: error)
            }
            request.imageCropAndScaleOption = .scaleFit
            self.visionRequests = [request]
            
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }
    
    // MARK: - Core Logic & Action
    @objc private func captureButtonTapped() {
        guard let pixelBuffer = self.latestPixelBuffer, !self.latestDetections.isEmpty else {
            showAlert(title: "No Detections", message: "Point the camera at some book spines and try again.")
            return
        }
        
        let detectionsToProcess = self.latestDetections
        let wholeImageID = UUID().uuidString
        
        // Serialize the bounding box detections into a JSON string.
        var boundingBoxesJSON: String = "[]"
        do {
            let jsonData = try JSONEncoder().encode(detectionsToProcess)
            if let jsonString = String(data: jsonData, encoding: .utf8) {
                boundingBoxesJSON = jsonString
            }
        } catch {
            print("Error encoding bounding boxes: \(error)")
        }
        
        // Update UI immediately.
        DispatchQueue.main.async {
            self.captureButton.isEnabled = false
            self.activityIndicator.startAnimating()
            self.imagesTakenCount += 1
            self.updateStatsLabel()
        }

        // Perform all processing and networking off the main thread.
        visionQueue.async { [weak self] in
            guard let self = self else { return }

            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
                self.handleUploadCompletion()
                return
            }
            let sourceImage = UIImage(cgImage: cgImage)
            
            // Upload the whole image WITH the bounding box JSON.
            NetworkService.shared.uploadWholeImage(image: sourceImage, wholeImageID: wholeImageID, boundingBoxesJSON: boundingBoxesJSON) { result in
                switch result {
                case .success:
                    print("Successfully uploaded whole image with ID: \(wholeImageID) and its bounding boxes.")
                case .failure(let error):
                    print("Failed to upload whole image: \(error.localizedDescription)")
                }
            }
            
            let modelContentRect = AVMakeRect(aspectRatio: self.bufferSize, insideRect: CGRect(origin: .zero, size: self.modelInputSize))
            let group = DispatchGroup()
            
            // Loop through detections, crop, and upload each one individually.
            for detection in detectionsToProcess {
                guard let croppedImage = self.crop(image: sourceImage, with: detection, modelContentRect: modelContentRect) else { continue }
                
                group.enter()
                NetworkService.shared.uploadCrop(image: croppedImage, wholeImageID: wholeImageID) { result in
                    switch result {
                    case .success:
                        self.cropsUploadedCount += 1
                        self.updateStatsLabel()
                    case .failure(let error):
                        print("Upload failed for a spine: \(error.localizedDescription)")
                    }
                    group.leave()
                }
            }

            // Once all individual crop uploads are complete, re-enable the UI.
            group.notify(queue: .main) {
                self.handleUploadCompletion()
            }
        }
    }
    
    private func handleUploadCompletion() {
        DispatchQueue.main.async {
            self.activityIndicator.stopAnimating()
            self.captureButton.isEnabled = true
        }
    }
    
    // MARK: - Vision & Drawing
    private func handleVisionRequest(request: VNRequest, error: Error?) {
        if let error = error {
            print("Vision request error: \(error.localizedDescription)")
            return
        }

        guard let results = request.results as? [VNCoreMLFeatureValueObservation],
              let multiArray = results.first?.featureValue.multiArrayValue else {
            return
        }
        
        let detectedBoxes = parseModelOutput(from: multiArray)
        let finalBoxes = nonMaxSuppression(boxes: detectedBoxes, iouThreshold: self.iouThreshold)
        
        self.latestDetections = finalBoxes
        
        DispatchQueue.main.async {
            self.clearBoundingBoxes()
            self.drawOrientedBoundingBoxes(finalBoxes)
        }
    }

    private func parseModelOutput(from multiArray: MLMultiArray) -> [OrientedBoundingBox] {
        var detectedBoxes: [OrientedBoundingBox] = []
        let detectionCount = multiArray.shape[2].intValue
        let totalCount = multiArray.count
        let pointer = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: totalCount)
        
        let cxPtr = pointer
        let cyPtr = cxPtr.advanced(by: detectionCount)
        let wPtr = cyPtr.advanced(by: detectionCount)
        let hPtr = wPtr.advanced(by: detectionCount)
        let confPtr = hPtr.advanced(by: detectionCount)
        let anglePtr = confPtr.advanced(by: detectionCount)
        
        for i in 0..<detectionCount {
            let confidence = confPtr[i]
            if confidence >= self.confidenceThreshold {
                let box = OrientedBoundingBox(
                    center: CGPoint(x: CGFloat(cxPtr[i]), y: CGFloat(cyPtr[i])),
                    size: CGSize(width: CGFloat(wPtr[i]), height: CGFloat(hPtr[i])),
                    angle: CGFloat(anglePtr[i]),
                    confidence: confidence
                )
                detectedBoxes.append(box)
            }
        }
        return detectedBoxes
    }

    private func drawOrientedBoundingBoxes(_ boxes: [OrientedBoundingBox]) {
        for box in boxes {
            let corners = getTransformedCorners(for: box)
            guard !corners.isEmpty else { continue }
            
            let path = UIBezierPath()
            path.move(to: corners[0])
            for i in 1..<corners.count {
                path.addLine(to: corners[i])
            }
            path.close()
            
            let shapeLayer = CAShapeLayer()
            shapeLayer.path = path.cgPath
            shapeLayer.fillColor = UIColor.red.withAlphaComponent(0.2).cgColor
            shapeLayer.strokeColor = UIColor.red.cgColor
            shapeLayer.lineWidth = 2.0
            
            self.boundingBoxLayer.addSublayer(shapeLayer)
        }
    }
    
    private func clearBoundingBoxes() {
        self.boundingBoxLayer.sublayers?.forEach { $0.removeFromSuperlayer() }
    }

    private func getTransformedCorners(for box: OrientedBoundingBox) -> [CGPoint] {
        let rotatedCorners = box.corners
        guard bufferSize != .zero else { return [] }

        let modelContentRect = AVMakeRect(aspectRatio: bufferSize, insideRect: CGRect(origin: .zero, size: modelInputSize))

        return rotatedCorners.map { point -> CGPoint in
            let normalizedX = (point.x - modelContentRect.origin.x) / modelContentRect.size.width
            let normalizedY = (point.y - modelContentRect.origin.y) / modelContentRect.size.height
            let normalizedPoint = CGPoint(x: normalizedX, y: normalizedY)
            return previewLayer.layerPointConverted(fromCaptureDevicePoint: normalizedPoint)
        }
    }
    
    // MARK: - Helper Methods
    private func updateStatsLabel() {
        DispatchQueue.main.async {
            self.statsLabel.text = "Images: \(self.imagesTakenCount) | Crops: \(self.cropsUploadedCount)"
        }
    }
    
    private func crop(image: UIImage, with box: OrientedBoundingBox, modelContentRect: CGRect) -> UIImage? {
        guard let sourceCGImage = image.cgImage else { return nil }
        
        let scaleX = image.size.width / modelContentRect.width
        let scaleY = image.size.height / modelContentRect.height
        
        let imageSpaceSize = CGSize(width: box.size.width * scaleX, height: box.size.height * scaleY)
        let centerInImage_BottomOrigin = CGPoint(x: (box.center.x - modelContentRect.origin.x) * scaleX, y: (box.center.y - modelContentRect.origin.y) * scaleY)
        let imageSpaceCenter = CGPoint(x: centerInImage_BottomOrigin.x, y: image.size.height - centerInImage_BottomOrigin.y)
        
        UIGraphicsBeginImageContextWithOptions(imageSpaceSize, false, image.scale)
        guard let context = UIGraphicsGetCurrentContext() else {
            UIGraphicsEndImageContext()
            return nil
        }
        
        context.translateBy(x: imageSpaceSize.width / 2, y: imageSpaceSize.height / 2)
        context.rotate(by: box.angle)
        context.translateBy(x: -imageSpaceCenter.x, y: -imageSpaceCenter.y)
        context.draw(sourceCGImage, in: CGRect(origin: .zero, size: image.size))
        
        let croppedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return croppedImage
    }

    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        self.present(alert, animated: true)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        self.latestPixelBuffer = pixelBuffer
        self.bufferSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        
        var requestOptions: [VNImageOption: Any] = [:]
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) {
            requestOptions = [.cameraIntrinsics: cameraIntrinsicData]
        }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: requestOptions)
        do {
            try imageRequestHandler.perform(self.visionRequests)
        } catch {
            print("Failed to perform Vision request: \(error)")
        }
    }
}

// MARK: - Geometry & NMS Helpers
// These functions live outside the ViewController class.
func nonMaxSuppression(boxes: [OrientedBoundingBox], iouThreshold: Float) -> [OrientedBoundingBox] {
    let sortedBoxes = boxes.sorted { $0.confidence > $1.confidence }
    
    var selectedBoxes: [OrientedBoundingBox] = []
    var active = [Bool](repeating: true, count: sortedBoxes.count)
    
    for i in 0..<sortedBoxes.count where active[i] {
        selectedBoxes.append(sortedBoxes[i])
        
        for j in (i + 1)..<sortedBoxes.count where active[j] {
            let iou = calculateRotatedIoU(boxA: sortedBoxes[i], boxB: sortedBoxes[j])
            if iou > iouThreshold {
                active[j] = false
            }
        }
    }
    return selectedBoxes
}

func calculateRotatedIoU(boxA: OrientedBoundingBox, boxB: OrientedBoundingBox) -> Float {
    let intersectionArea = polygonIntersectionArea(polygonA: boxA.corners, polygonB: boxB.corners)
    if intersectionArea <= 0 { return 0 }
    
    let areaA = boxA.size.width * boxA.size.height
    let areaB = boxB.size.width * boxB.size.height
    let unionArea = areaA + areaB - intersectionArea
    
    return Float(intersectionArea / unionArea)
}

private func polygonIntersectionArea(polygonA: [CGPoint], polygonB: [CGPoint]) -> CGFloat {
    var clippedPolygon = polygonA
    for i in 0..<polygonB.count {
        let edgeStart = polygonB[i]
        let edgeEnd = polygonB[(i + 1) % polygonB.count]
        clippedPolygon = clipPolygon(subjectPolygon: clippedPolygon, clipEdgeStart: edgeStart, clipEdgeEnd: edgeEnd)
        if clippedPolygon.isEmpty { return 0 }
    }
    return polygonArea(points: clippedPolygon)
}

private func clipPolygon(subjectPolygon: [CGPoint], clipEdgeStart: CGPoint, clipEdgeEnd: CGPoint) -> [CGPoint] {
    var outputList: [CGPoint] = []
    guard !subjectPolygon.isEmpty else { return [] }
    var s = subjectPolygon.last!
    
    for e in subjectPolygon {
        let sInside = isInside(point: s, edgeStart: clipEdgeStart, edgeEnd: clipEdgeEnd)
        let eInside = isInside(point: e, edgeStart: clipEdgeStart, edgeEnd: clipEdgeEnd)
        
        if eInside {
            if !sInside, let intersection = intersection(s, e, clipEdgeStart, clipEdgeEnd) {
                outputList.append(intersection)
            }
            outputList.append(e)
        } else if sInside, let intersection = intersection(s, e, clipEdgeStart, clipEdgeEnd) {
            outputList.append(intersection)
        }
        s = e
    }
    return outputList
}

private func isInside(point: CGPoint, edgeStart: CGPoint, edgeEnd: CGPoint) -> Bool {
    return (edgeEnd.x - edgeStart.x) * (point.y - edgeStart.y) > (edgeEnd.y - edgeStart.y) * (point.x - edgeStart.x)
}

private func intersection(_ s: CGPoint, _ e: CGPoint, _ cs: CGPoint, _ ce: CGPoint) -> CGPoint? {
    let dc = CGPoint(x: cs.x - ce.x, y: cs.y - ce.y)
    let dp = CGPoint(x: s.x - e.x, y: s.y - e.y)
    let n1 = cs.x * ce.y - cs.y * ce.x
    let n2 = s.x * e.y - s.y * e.x
    let denominator = dc.x * dp.y - dc.y * dp.x
    
    if abs(denominator) < 1e-6 { return nil }
    
    let x = (n1 * dp.x - n2 * dc.x) / denominator
    let y = (n1 * dp.y - n2 * dc.y) / denominator
    return CGPoint(x: x, y: y)
}

private func polygonArea(points: [CGPoint]) -> CGFloat {
    guard points.count > 2 else { return 0 }
    var area: CGFloat = 0.0
    var j = points.count - 1
    for i in 0..<points.count {
        area += (points[j].x + points[i].x) * (points[j].y - points[i].y)
        j = i
    }
    return abs(area / 2.0)
}
